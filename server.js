require('dotenv').config();
const express = require('express');
const multer  = require('multer');
const fs      = require('fs');
const path    = require('path');
const cors    = require('cors');
const http    = require('http');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// ─── OLLAMA CONFIG ────────────────────────────────────────────
const OLLAMA_HOST  = process.env.OLLAMA_HOST  || 'localhost';
const OLLAMA_PORT  = parseInt(process.env.OLLAMA_PORT  || '11434', 10);
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.2';

console.log(`[Ollama] host=${OLLAMA_HOST}:${OLLAMA_PORT}  model=${OLLAMA_MODEL}`);

// ─── DIRECTORIES ─────────────────────────────────────────────
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const dataDir  = path.join(__dirname, 'data');
const dataFile = path.join(dataDir, 'results.json');
if (!fs.existsSync(dataDir))  fs.mkdirSync(dataDir);
if (!fs.existsSync(dataFile)) fs.writeFileSync(dataFile, JSON.stringify({ results: [] }, null, 2));

const resumeStorage = multer.diskStorage({
  destination: uploadDir,
  filename:    (req, file, cb) => cb(null, Date.now() + '-' + file.originalname),
});
const resumeUpload = multer({ storage: resumeStorage });

// ─── CORE: callOllama (streaming) ────────────────────────────
// Uses /api/chat with stream:true.
// WHY STREAMING: Without it, Ollama generates the entire response
// before sending a single byte. On slow/CPU machines this means
// 60+ seconds of silence then a timeout. With streaming, each token
// arrives immediately, keeping the connection alive and giving us
// an idle-chunk watchdog instead of one hard deadline.
function callOllama(userContent, {
  temperature = 0.2,
  maxTokens   = 250,
  system      = '',
  timeoutMs   = 90000,
} = {}) {
  return new Promise((resolve, reject) => {
    const messages = [];
    if (system) messages.push({ role: 'system', content: system });
    messages.push({ role: 'user', content: userContent });

    const body = JSON.stringify({
      model:   OLLAMA_MODEL,
      messages,
      stream:  true,
      options: {
        temperature,
        num_predict:    maxTokens,
        top_p:          0.9,
        repeat_penalty: 1.1,
        stop:           ['```', '\n\n\n'],
      },
    });

    const reqOptions = {
      hostname: OLLAMA_HOST,
      port:     OLLAMA_PORT,
      path:     '/api/chat',
      method:   'POST',
      headers: {
        'Content-Type':   'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    };

    let fullText    = '';
    let lastChunkAt = Date.now();
    let dead        = false;

    // Idle watchdog: if no chunk arrives for 25s, give up
    const idleTimer = setInterval(() => {
      if (Date.now() - lastChunkAt > 25000) {
        dead = true;
        clearInterval(idleTimer);
        req.destroy();
        reject(new Error(
          'Ollama stopped mid-stream (25s idle). ' +
          'Try a smaller model: ollama pull phi3'
        ));
      }
    }, 5000);

    const req = http.request(reqOptions, (res) => {
      let buf = '';

      res.on('data', chunk => {
        if (dead) return;
        lastChunkAt = Date.now();
        buf += chunk.toString();

        // Each line is one JSON object in the streaming protocol
        const lines = buf.split('\n');
        buf = lines.pop(); // keep incomplete last line in buffer
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const obj = JSON.parse(trimmed);
            if (obj?.message?.content) fullText += obj.message.content;
          } catch (_) {}
        }
      });

      res.on('end', () => {
        clearInterval(idleTimer);
        if (dead) return;
        // Flush any remaining buffered content
        if (buf.trim()) {
          try {
            const obj = JSON.parse(buf.trim());
            if (obj?.message?.content) fullText += obj.message.content;
          } catch (_) {}
        }
        resolve(fullText.trim());
      });

      res.on('error', err => { clearInterval(idleTimer); reject(err); });
    });

    req.on('error', err => {
      clearInterval(idleTimer);
      if (err.code === 'ECONNREFUSED') {
        reject(new Error(
          'Ollama is not running.\n' +
          '  1. ollama serve\n' +
          `  2. ollama pull ${OLLAMA_MODEL}`
        ));
      } else {
        reject(err);
      }
    });

    // Hard cap — safety net if the idle timer somehow misses
    req.setTimeout(timeoutMs, () => {
      clearInterval(idleTimer);
      req.destroy();
      reject(new Error(
        `Ollama exceeded ${timeoutMs / 1000}s total. ` +
        'Try: ollama pull phi3'
      ));
    });

    req.write(body);
    req.end();
  });
}

// ─── HELPER: robustly extract JSON from LLM output ───────────
// Models often add prose, reasoning tags, or markdown fences.
// This finds the outermost { } block reliably.
function extractJSON(raw) {
  let s = raw
    .replace(/^```json\s*/im, '')
    .replace(/^```\s*/im,     '')
    .replace(/```\s*$/m,      '')
    .trim();

  const start = s.indexOf('{');
  const end   = s.lastIndexOf('}');
  if (start !== -1 && end > start) s = s.slice(start, end + 1);

  return JSON.parse(s);
}

// ─── WARM-UP: load model into RAM at server start ─────────────
// First Ollama request takes 10-30s to load the model.
// Doing it at startup means the first real user request is instant.
async function warmUpModel() {
  try {
    console.log(`[Ollama] Loading model "${OLLAMA_MODEL}" into memory...`);
    await callOllama('hi', {
      maxTokens: 3,
      system:    'Say hi.',
      timeoutMs: 60000,
    });
    console.log('[Ollama] Model ready ✓');
  } catch (err) {
    console.warn(`[Ollama] Warm-up skipped (will load on first request): ${err.message}`);
  }
}

// ─── HEALTH CHECK ────────────────────────────────────────────
app.get('/api/health', async (req, res) => {
  try {
    await callOllama('hi', { maxTokens: 3, system: 'Say hi.', timeoutMs: 15000 });
    res.json({ status: 'ok', model: OLLAMA_MODEL, host: `${OLLAMA_HOST}:${OLLAMA_PORT}` });
  } catch (err) {
    res.status(503).json({ status: 'error', message: err.message, model: OLLAMA_MODEL });
  }
});

// ─── RESUME PARSE ────────────────────────────────────────────
app.post('/api/parse-resume', resumeUpload.single('resume'), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'no file' });

  let resumeText = '';
  try { resumeText = fs.readFileSync(file.path, 'utf8').slice(0, 2000); } catch (_) {}

  const fallback = {
    name:       path.basename(file.originalname).replace(/\.[^.]+$/, ''),
    skills:     [],
    experience: '',
    education:  '',
  };

  if (!resumeText) return res.json({ parsed: fallback, profileId: Date.now().toString() });

  try {
    const raw = await callOllama(
      `Extract from resume. Return ONLY this JSON filled in:\n` +
      `{"name":"","skills":[],"experience":"","education":""}\n\n` +
      resumeText.slice(0, 1500),
      { maxTokens: 180, system: 'Return ONLY a JSON object. No markdown.', timeoutMs: 45000 }
    );
    const parsed = extractJSON(raw);
    res.json({ parsed, profileId: Date.now().toString() });
  } catch (err) {
    console.error('[parse-resume]', err.message);
    res.json({ parsed: fallback, profileId: Date.now().toString() });
  }
});

// ─── LEVEL-AWARE QUESTION BANKS (fallback) ───────────────────
const LEVEL_BANKS = {
  fresher: {
    'Software Engineer': [
      'What is the difference between an array and a linked list?',
      'Explain what Object-Oriented Programming means to you.',
      'What is the difference between a stack and a queue?',
      'Can you explain what version control is and why it matters?',
      'What is the difference between HTTP and HTTPS?',
      'What does it mean for code to be "readable"?',
      'Have you worked on any personal or college projects? Tell me about one.',
      'What programming languages are you comfortable with and why?',
      'How do you approach debugging when your code does not work?',
      'What are your strengths and areas you want to improve as a developer?',
      'Describe a time you had a conflict with a teammate and how you handled it.',
      'Why did you choose this field and what excites you about it?',
    ],
    'Data Scientist': [
      'What is the difference between supervised and unsupervised learning?',
      'Can you explain what a training set and test set are?',
      'What is overfitting and how would you detect it?',
      'Explain mean, median, and mode — when would you use each?',
      'What Python libraries have you used for data analysis?',
      'Describe a data project you worked on in college or personally.',
      'What is a confusion matrix?',
      'Why is data cleaning important?',
    ],
    generic: [
      'Tell me about yourself and what drew you to this field.',
      'What are your strongest technical skills?',
      'Describe a college project you are proud of.',
      'How do you handle a task you have never done before?',
      'What does good teamwork look like to you?',
      'Where do you want to be professionally in 2 years?',
      'How do you manage your time when working on multiple assignments?',
      'What are your hobbies outside of work or study?',
    ],
  },
  junior: {
    'Software Engineer': [
      'Walk me through a feature you built end-to-end in your first job.',
      'How do you write unit tests and why are they important?',
      'Explain the difference between SQL and NoSQL databases.',
      'What is REST and what makes a good RESTful API?',
      'How do you handle merge conflicts in Git?',
      'What is the difference between authentication and authorization?',
      'How would you improve the performance of a slow-loading web page?',
      'Describe a production bug you fixed. What was your process?',
      'What is Big O notation? Give an example.',
      'How do you keep up with new technologies?',
      'Tell me about a time you received critical feedback on your code.',
    ],
    'Data Scientist': [
      'Walk me through a model you built and deployed.',
      'How do you handle missing data in a dataset?',
      'What is the difference between precision and recall?',
      'How do you validate a machine learning model?',
      'What is feature engineering? Give an example you have done.',
      'How do you communicate model results to a non-technical audience?',
    ],
    generic: [
      'Tell me about yourself and your professional experience so far.',
      'Describe a challenging project and how you overcame a technical obstacle.',
      'How do you prioritize tasks when you have multiple deadlines?',
      'Tell me about a time you disagreed with a decision and what you did.',
      'What is one technical skill you have improved recently?',
      'Where do you see yourself in 3 years?',
    ],
  },
  mid: {
    'Software Engineer': [
      'How would you design a URL shortener that handles 100 million requests per day?',
      'Describe your approach to refactoring a large legacy codebase.',
      'How do you ensure code quality across a team — beyond just code reviews?',
      'Explain CAP theorem and how it influenced a design decision you made.',
      'How do you approach database schema design for a greenfield product?',
      'Walk me through a system you designed that had to scale.',
      'What is your approach to incident management in production?',
      'How do you mentor junior developers effectively?',
      'Describe a time a project failed — what did you learn and change?',
      'How do you evaluate whether to build a feature in-house or use a third-party service?',
      'What trade-offs have you made between technical debt and shipping speed?',
    ],
    'Data Scientist': [
      'Describe a model you built that had a real measurable business impact.',
      'How do you monitor models in production for data drift?',
      'Walk me through your approach to A/B testing a new ML feature.',
      'How do you handle class imbalance in a classification problem?',
      'Describe your MLOps workflow — from training to deployment.',
      'How do you work with product and engineering teams on AI features?',
    ],
    generic: [
      'Tell me about yourself and the most impactful project you have led.',
      'Describe a situation where you influenced a technical decision across teams.',
      'How do you manage technical debt while still shipping features?',
      'Tell me about a time you had to lead without formal authority.',
      'How do you approach cross-functional communication on complex projects?',
      'Where do you see yourself in 5 years?',
    ],
  },
  senior: {
    'Software Engineer': [
      'Walk me through the most complex distributed system you have architected.',
      'How do you define and uphold engineering standards across multiple teams?',
      'Describe your approach to technology evaluation and adoption at scale.',
      'How do you handle a situation where a team consistently misses delivery timelines?',
      'How do you balance innovation with stability in a mature product?',
      'Describe a time you had to make a high-stakes technical call with incomplete information.',
      'How do you think about engineering culture and what have you done to improve it?',
      'What is your philosophy on technical documentation and why?',
      'Describe a significant architecture decision you reversed — what did you learn?',
      'How do you evaluate and retain technical talent on your team?',
      'How do you influence company strategy from an engineering perspective?',
    ],
    'Data Scientist': [
      'How have you built a data science function or practice from scratch?',
      'Describe a time your model recommendations were rejected by leadership — how did you respond?',
      'How do you build long-term trust with business stakeholders around AI?',
      'What is your approach to ethical AI and bias detection in production systems?',
      'How do you set the technical roadmap for a data science team?',
    ],
    generic: [
      'Walk me through your leadership philosophy and how it has evolved.',
      'Describe the most difficult personnel decision you have made and what you learned.',
      'How do you approach building high-trust, high-performance teams?',
      'Tell me about a time you changed the strategic direction of a product or team.',
      'How do you balance being a technical leader with management responsibilities?',
      'What legacy do you want to leave at the companies you join?',
      'Describe how you have influenced organizational culture.',
    ],
  },
};

function normLevel(level) {
  const l = (level || '').toLowerCase();
  if (l.includes('fresh') || l.includes('0-1') || l.includes('intern') || l.includes('entry')) return 'fresher';
  if (l.includes('junior') || l.includes('1-3') || l.includes('1-2')) return 'junior';
  if (l.includes('senior') || l.includes('5+') || l.includes('6+') || l.includes('7+') || l.includes('lead') || l.includes('staff') || l.includes('principal')) return 'senior';
  return 'mid'; // default mid
}

function makeLevelQuestions(role, level, count) {
  const lk   = normLevel(level);
  const bank  = LEVEL_BANKS[lk] || LEVEL_BANKS.mid;
  const roleQ = bank[role] || bank['Software Engineer'] || [];
  const genQ  = bank.generic || [];

  const opener = {
    fresher: `Tell me about yourself and what inspired you to pursue a career as a ${role}.`,
    junior:  `Tell me about yourself and your experience so far as a ${role}.`,
    mid:     `Tell me about yourself and the most impactful project you have led as a ${role}.`,
    senior:  `Walk me through your career journey and what has driven your growth as a ${role}.`,
  }[lk];

  const pool = [...roleQ, ...genQ].sort(() => Math.random() - 0.5);
  const qs   = [opener, ...pool].slice(0, count);
  return qs;
}

// ─── GENERATE INTERVIEW QUESTIONS ────────────────────────────
// POST /api/generate-questions  { role, level, resume, count }
app.post('/api/generate-questions', async (req, res) => {
  const { role = 'Software Engineer', level = 'Mid-level', resume = '', count = 8 } = req.body;
  const lk = normLevel(level);

  const levelDesc = {
    fresher: 'a fresher (0-1 years exp). Focus on: basic concepts, college projects, fundamental knowledge, professional ethics, communication, and learning mindset. Avoid complex system design.',
    junior:  'a junior developer (1-3 years exp). Focus on: practical experience, code quality, basic architecture, debugging, teamwork, and growth. Mix basic and intermediate technical questions.',
    mid:     'a mid-level engineer (3-6 years exp). Focus on: system design, architectural decisions, code quality at scale, cross-team collaboration, mentorship, and technical trade-offs.',
    senior:  'a senior/lead engineer (6+ years exp). Focus on: complex system architecture, engineering leadership, team culture, strategic technical decisions, organizational influence, and people management.',
  }[lk];

  const hasResume  = resume && resume.trim().length > 30;
  const resumeSnip = hasResume ? resume.trim().slice(0, 1000) : '';

  const system  = `You are an expert technical interviewer. Reply ONLY with valid JSON: {"questions":["q1","q2",...]}. No markdown. No numbering inside question text. No extra keys.`;
  const userMsg = `Generate exactly ${count} interview questions for ${role} — ${levelDesc}\n` +
    (hasResume ? `Candidate resume context: ${resumeSnip}\n` : '') +
    `Question order:\n` +
    `- Q1: Opening/intro question appropriate for ${lk} level\n` +
    `- Q2-Q${Math.max(2, count-3)}: Technical questions calibrated for ${lk} level\n` +
    `- Q${count-2}: Behavioural (situation/challenge/conflict)\n` +
    `- Q${count-1}: Career goals / where do you see yourself\n` +
    `- Q${count}: Professional ethics / teamwork / culture fit\n` +
    `JSON:`;

  try {
    const raw  = await callOllama(userMsg, { temperature: 0.5, maxTokens: 700, system, timeoutMs: 90000 });
    const data = extractJSON(raw);

    if (!Array.isArray(data.questions) || data.questions.length < 3) {
      throw new Error('Too few questions returned: ' + JSON.stringify(data));
    }

    data.questions = data.questions.slice(0, count);
    console.log(`[generate-questions] ${data.questions.length} Qs for ${role} (${level}/${lk})`);
    res.json({ questions: data.questions, source: 'ollama' });
  } catch (err) {
    console.error('[generate-questions] Ollama failed, using level banks:', err.message);
    // Always fall back to level-aware banks
    const qs = makeLevelQuestions(role, level, count);
    res.json({ questions: qs, source: 'level_bank' });
  }
});

// ─── EVALUATE ANSWER ─────────────────────────────────────────
// POST /api/evaluate  { question, answer, role, level }
app.post('/api/evaluate', async (req, res) => {
  const { question, answer, role = 'Software Engineer', level = 'Mid-level' } = req.body;

  if (!answer || answer.trim().length < 5) {
    return res.json({
      scores:       { tech: 0, comm: 0, conf: 0 },
      feedback:     'No meaningful answer provided.',
      strengths:    [],
      improvements: ['Speak a full answer before clicking Next Question.'],
    });
  }

  const system =
    'You are a FAIR, BALANCED interview evaluator — like a real human hiring manager. Reply ONLY with this exact JSON (no markdown):\n' +
    '{"scores":{"tech":0,"comm":0,"conf":0},"feedback":"Two honest sentences.","strengths":[],"improvements":[],"expectedAnswer":""}\n' +
    'SCORING GUIDE — use the FULL 0-100 range realistically:\n' +
    '- tech: No answer/completely wrong = 0-20. Vague/partial = 21-40. Basic correct understanding = 41-60. Clear correct with detail = 61-75. Excellent with examples/depth = 76-100.\n' +
    '- comm: 1 sentence/incoherent = 0-20. Too brief but understandable = 21-40. Reasonable structure = 41-60. Well structured = 61-75. STAR/clear narrative with examples = 76-100.\n' +
    '- conf: No answer/admitting ignorance = 0-20. Very hesitant = 21-40. Moderate confidence = 41-60. Assertive and clear = 61-75. Confident with specific examples = 76-100.\n' +
    'CALIBRATION: A decent fresher answer should score 45-65. A good answer should score 65-80. An excellent answer should score 80-95. Be fair — do not penalise candidates for minor imperfections.\n' +
    '- strengths: List 1-3 genuine strengths. If something was done well, acknowledge it.\n' +
    '- improvements: Always give 2-3 specific, actionable improvements tailored to THIS answer.\n' +
    '- expectedAnswer: Write the ACTUAL 3-5 sentence answer a top candidate would give for THIS SPECIFIC QUESTION. Include real technical content, specific frameworks/tools/approaches, and a concrete outcome.';

  const answerSnip = answer.trim().slice(0, 500);
  const userMsg    =
    `Role: ${role} (${level})\n` +
    `Question: ${question}\n` +
    `Candidate Answer (score this honestly — be strict): "${answerSnip}"\n` +
    `Return strict JSON evaluation:`;
  try {
    const raw = await callOllama(userMsg, {
      temperature: 0.1,
      maxTokens:   200,
      system,
      timeoutMs:   60000,
    });

    const parsed = extractJSON(raw);
    const clamp  = v => Math.max(0, Math.min(100, Math.round(Number(v) || 0)));

    parsed.scores = {
      tech: clamp(parsed.scores?.tech),
      comm: clamp(parsed.scores?.comm),
      conf: clamp(parsed.scores?.conf),
    };
    parsed.strengths    = (Array.isArray(parsed.strengths)    ? parsed.strengths    : []).slice(0, 3);
    parsed.improvements = (Array.isArray(parsed.improvements) ? parsed.improvements : []).slice(0, 3);
    parsed.feedback     = String(parsed.feedback || '').slice(0, 300);

    console.log(`[evaluate] Ollama OK tech:${parsed.scores.tech} comm:${parsed.scores.comm} conf:${parsed.scores.conf}`);
    return res.json(parsed);
  } catch (err) {
    console.warn('[evaluate] Ollama failed, trying Claude API fallback:', err.message);
  }

  // Fallback: Claude API via direct HTTP
  try {
    const claudeBody = JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 400,
      system: 'You are a FAIR, BALANCED interview evaluator — like a real human hiring manager. Reply ONLY with valid JSON, no markdown.\n' +
        'Schema: {"scores":{"tech":0,"comm":0,"conf":0},"feedback":"Two honest sentences.","strengths":[],"improvements":[],"expectedAnswer":""}\n' +
        'SCORING GUIDE — use the FULL 0-100 range realistically:\n' +
        '- tech: No answer/completely wrong = 0-20. Vague/partial = 21-40. Basic correct = 41-60. Clear+detailed = 61-75. Excellent with examples = 76-100.\n' +
        '- comm: 1 sentence/incoherent = 0-20. Too brief = 21-40. Reasonable = 41-60. Well structured = 61-75. STAR/clear narrative = 76-100.\n' +
        '- conf: No answer = 0-20. Very hesitant = 21-40. Moderate = 41-60. Assertive = 61-75. Confident with examples = 76-100.\n' +
        'CALIBRATION: A decent fresher answer = 45-65. A good answer = 65-80. Excellent = 80-95. Be fair, not harsh.\n' +
        '- expectedAnswer: The ACTUAL answer a top candidate gives for THIS question — specific, technical, real content.',
      messages: [{ role: 'user', content: `Role: ${role} (${level})\nQuestion: ${question}\nCandidate said: "${answerSnip}"\n\nEvaluate strictly and return JSON:` }],
    });

    const claudeResult = await new Promise((resolve, reject) => {
      const opts = {
        hostname: 'api.anthropic.com',
        path: '/v1/messages',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
          'Content-Length': Buffer.byteLength(claudeBody),
        },
      };
      const https = require('https');
      let body = '';
      const r = https.request(opts, (resp) => {
        resp.on('data', c => body += c);
        resp.on('end', () => { try { resolve(JSON.parse(body)); } catch(e) { reject(e); } });
      });
      r.setTimeout(15000, () => { r.destroy(); reject(new Error('Claude API timeout')); });
      r.on('error', reject);
      r.write(claudeBody);
      r.end();
    });

    const text = claudeResult?.content?.[0]?.text || '';
    const parsed = extractJSON(text);
    const clamp  = v => Math.max(0, Math.min(100, Math.round(Number(v) || 50)));
    parsed.scores = {
      tech: clamp(parsed.scores?.tech),
      comm: clamp(parsed.scores?.comm),
      conf: clamp(parsed.scores?.conf),
    };
    parsed.strengths    = (Array.isArray(parsed.strengths)    ? parsed.strengths    : []).slice(0, 3);
    parsed.improvements = (Array.isArray(parsed.improvements) ? parsed.improvements : []).slice(0, 3);
    parsed.feedback     = String(parsed.feedback || '').slice(0, 300);

    console.log(`[evaluate] Claude API OK tech:${parsed.scores.tech} comm:${parsed.scores.comm}`);
    return res.json(parsed);

  } catch (err2) {
    console.error('[evaluate] Both Ollama and Claude API failed:', err2.message);

    // Last resort: smart heuristic fallback
    const words   = answer.trim().split(/\s+/).length;
    const fillers = (answer.toLowerCase().match(/\b(um|uh|like|you know|basically|sort of)\b/g) || []).length;
    const hasTech = /api|function|system|database|deploy|algorithm|framework|design|test|code|architecture|implement|solution|approach|method|process/i.test(answer);
    const hasStar = /situation|task|action|result|i led|i built|i fixed|i solved|i worked|i managed|i developed|i created|i designed/i.test(answer);
    const hasDetail = words > 80 && (hasTech || hasStar);
    const jitter  = () => Math.floor((Math.random() - 0.5) * 8);

    // Realistic base scores: short/vague answers get 20-40, decent answers 40-65, detailed answers 60-80
    const techBase = hasTech ? (hasDetail ? 62 : 45) : (words > 30 ? 28 : 12);
    const commBase = hasStar ? (words > 80 ? 65 : 50) : (words > 50 ? 42 : 22);
    const confBase = fillers < 2 ? (words > 60 ? 60 : 42) : (fillers < 5 ? 38 : 22);

    return res.json({
      scores: {
        tech: Math.min(100, Math.max(0, techBase + jitter())),
        comm: Math.min(100, Math.max(0, commBase + jitter())),
        conf: Math.min(100, Math.max(0, confBase + jitter())),
      },
      feedback: words < 20
        ? 'Answer was too brief — aim for at least 60-80 words with clear reasoning.'
        : hasTech
          ? 'Good technical content. Structure your answer more clearly using STAR or a logical flow.'
          : 'Add specific technical details, examples, or metrics to strengthen this answer.',
      strengths:    words > 60 ? ['Good response length'] : [],
      improvements: fillers > 3
        ? ['Reduce filler words (um, uh, like) — pause instead']
        : ['Use the STAR method: Situation, Task, Action, Result'],
    });
  }
});
//
// ─── SAVE / FETCH RESULTS ────────────────────────────────────
app.post('/api/results', (req, res) => {
  const store = JSON.parse(fs.readFileSync(dataFile, 'utf8'));
  const entry = { id: Date.now().toString(), timestamp: new Date().toISOString(), payload: req.body };
  store.results.unshift(entry);
  if (store.results.length > 200) store.results = store.results.slice(0, 200);
  fs.writeFileSync(dataFile, JSON.stringify(store, null, 2));
  res.json({ ok: true, id: entry.id });
});

app.get('/api/history', (req, res) => {
  const store = JSON.parse(fs.readFileSync(dataFile, 'utf8'));
  res.json(store.results);
});

app.get('/api/questions', (req, res) => {
  res.json({ questions: [
    { id: 1, type: 'behavioral', text: 'Tell me about a time you solved a difficult problem.' },
    { id: 2, type: 'technical',  text: 'Explain the difference between event loop and threads.' },
    { id: 3, type: 'hr',         text: 'Why are you interested in this role?' },
  ]});
});

// ─── RF CONFIDENCE PROXY ─────────────────────────────────────
// Forwards to the Python Flask service on RF_PORT (default 5050).
// Returns a graceful fallback if Flask is offline so the interview
// always continues even without the Python backend.
const RF_HOST = process.env.RF_HOST || 'localhost';
const RF_PORT = parseInt(process.env.RF_PORT || '5050', 10);

app.post('/api/rf-confidence', async (req, res) => {
  const { text = '', num_questions = 1, face_detected, eye_contact, posture_score } = req.body;

  if (!text.trim()) {
    return res.status(400).json({ success: false, error: 'text is required' });
  }

  const payload = JSON.stringify({
    text,
    num_questions: Number(num_questions) || 1,
    face_detected: face_detected !== undefined ? Number(face_detected) : 1,
    eye_contact:   eye_contact   !== undefined ? Number(eye_contact)   : 0.65,
    posture_score: posture_score !== undefined ? Number(posture_score) : 0.65,
  });

  try {
    const result = await new Promise((resolve, reject) => {
      const options = {
        hostname: RF_HOST,
        port:     RF_PORT,
        path:     '/predict',
        method:   'POST',
        headers: {
          'Content-Type':   'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
      };

      let body = '';
      const req2 = http.request(options, (r) => {
        r.on('data', chunk => { body += chunk; });
        r.on('end',  ()    => {
          try { resolve(JSON.parse(body)); }
          catch (e) { reject(new Error('Bad JSON from RF service: ' + body.slice(0, 120))); }
        });
        r.on('error', reject);
      });

      req2.setTimeout(8000, () => { req2.destroy(); reject(new Error('RF service timeout')); });
      req2.on('error', err => {
        if (err.code === 'ECONNREFUSED') reject(new Error('RF Flask service offline (ECONNREFUSED)'));
        else reject(err);
      });
      req2.write(payload);
      req2.end();
    });

    res.json(result);

  } catch (err) {
    console.warn('[rf-confidence] Flask unavailable, using heuristic fallback:', err.message);

    // ── Heuristic fallback (same logic as scoreHeuristic in the frontend) ──
    const words   = text.trim().split(/\s+/).filter(w => w.length > 1);
    const wc      = words.length;
    const fillers = (text.toLowerCase().match(/\b(um|uh|like|you know|basically|sort of|i mean)\b/g) || []).length;
    const fr      = fillers / Math.max(wc, 1);
    const jit     = () => Math.floor((Math.random() - 0.5) * 10);
    const cl      = v => Math.max(10, Math.min(100, Math.round(v)));

    let base = 55;
    if (wc > 150) base += 15; else if (wc < 40) base -= 18;
    if (fr > 0.08) base -= 14; else if (fr < 0.02) base += 12;

    const label      = base >= 75 ? 'High' : base >= 48 ? 'Medium' : 'Low';
    const conf_score = cl(base + jit());

    res.json({
      success:       true,
      label,
      conf_score,
      probabilities: { Low: label === 'Low' ? 0.7 : 0.1, Medium: label === 'Medium' ? 0.7 : 0.15, High: label === 'High' ? 0.7 : 0.1 },
      text_analysis: { total_words: wc, total_fillers: fillers, filler_ratio: +fr.toFixed(4) },
      feedback: {
        label,
        tip:          label === 'Low' ? 'Expand answers and reduce filler words.' : label === 'Medium' ? 'Add concrete examples to boost your score.' : 'Excellent — keep it up!',
        strengths:    wc > 100 ? ['Good answer length'] : [],
        improvements: fr > 0.05 ? ['Reduce filler words (um, uh, like)'] : [],
      },
      _source: 'heuristic_fallback',
    });
  }
});

// ─── START ───────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  console.log(`\nSwaKrut (Ollama) → http://localhost:${PORT}`);
  console.log(`Ollama endpoint  → http://${OLLAMA_HOST}:${OLLAMA_PORT}`);
  console.log(`Active model     → ${OLLAMA_MODEL}\n`);
  warmUpModel();
});