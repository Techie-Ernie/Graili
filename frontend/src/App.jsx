import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MODEL = "gemini-2.5-flash-lite";

const defaultSubjects = [];
const defaultCategories = [];
const questionTypeOptions = ["exam", "understanding"];
const CANONICAL_CATEGORY = "GCE 'A' Levels";

const fallbackYears = Array.from({ length: 10 }, (_, index) => {
  const year = new Date().getFullYear() - index;
  return year;
});

const basePrompt = `
You are an information extraction engine. You MUST NOT generate any content that is not present verbatim in the input.

TASK
Extract ONLY the actual exam sub-questions from the given paper text (ignore instructions, headers, source lines, extracts, figures, totals, and any non-question content). Classify each extracted sub-question into:
- "exam": sub-questions that have an explicit mark allocation in square brackets, attached per rules below
- "understanding": sub-questions that do NOT have an explicit mark allocation attached per rules below

CRITICAL: NO HALLUCINATIONS
- Output ONLY questions that appear verbatim in the input text.
- DO NOT paraphrase, summarize, or invent any questions.
- DO NOT create “understanding” questions unless they are verbatim in the input.
- If there are no understanding questions, "understanding" MUST be [].

DEFINITION OF QUESTION
A “question” is a line/block that asks the candidate to do something (e.g., Explain/Discuss/Describe/Analyse/Compare/Evaluate/With reference to..., etc.), including subparts like (a), (b), (c)(i), (ii), etc.

EXCLUSIONS (NOT QUESTIONS)
Do NOT extract:
- General instructions (e.g., “Answer all questions”, “You are required to answer…”)
- Section instructions (e.g., “One or two of your three chosen questions…”)
- Totals lines like “[Total: 30]”
- Any bracketed text that is NOT a mark allocation (e.g., quotes like “free trade [but] ...”)
- Source citations and headers

MARK ASSOCIATION RULE (STRICT + MECHANICAL)
A sub-question has marks ONLY if there is a square-bracketed integer mark like [2], [4], [8], [10], [15] that is:
1) On the same line as the sub-question, OR
2) On the immediately following NON-EMPTY line (after trimming whitespace).

A square-bracketed value is NOT a mark allocation if it matches:
- [Total: ...] (case-insensitive)
- any bracketed word/phrase containing letters (e.g., [but], [Total], [Source])
Only pure integers inside brackets count as marks: [number]

If a pure-integer mark is found by rule (1) or (2), append it to the end of the question text separated by a single space, exactly as written (e.g., "... [10]").
If no such mark is attached, DO NOT add any brackets.

THEME & SUBTHEME MAPPING (STRICT)

- You MUST assign a syllabus chapter AND subchapter number (e.g. 2.2.2) to EVERY extracted question.
- Do NOT output "Unknown".
- Do NOT skip chapter assignment.
- Use the MOST SPECIFIC applicable subchapter number from the syllabus.
- The chapter value MUST exactly match one of the provided subtopics (code + title).

Theme format MUST be:
"<syllabus_number> <syllabus_chapter_title>"

Examples:
- "2.2.2 Inflation and its causes"
- "1.3.1 Price controls"
- "3.1.4 Trade protectionism"


OUTPUT FORMAT (STRICT JSON ONLY)
Return JSON ONLY in this exact structure:
{
  "exam": [
    { "chapter": "<chapter>", "question": "<verbatim question text with attached mark if present>", "marks": <number of marks allocated>}
  ],
  "understanding": [
    { "chapter": "<chapter>", "question": "<verbatim question text with NO marks>" }
  ]
}

FINAL VALIDATION (MANDATORY, DO NOT OUTPUT)
- Every "exam" question MUST end with a pure-integer bracket mark like [10].
- No "understanding" question may contain '[' or ']'.
- Do not output any item that is not verbatim from the input.

SYLLABUS + TEXT:
`;

const subtopicPrompt = `
You are a strict information extraction engine. Extract syllabus subject and subtopics from the input text.

Rules:
- A subject is the name of the subject and its subject code e.g. Economics (Syllabus 9750) and Physics (Syllabus 6091) 
- A subtopic is a dotted numeric code followed by a title. Examples: "1.1 Functions", "2.3.4 Elasticity".
- If a line is only a dotted numeric code, use the next non-empty line as its title.
- Ignore any non-subtopic content (prefaces, assessment objectives, admin sections).
- Do not invent or paraphrase.
- No duplicates by code.

Output JSON ONLY in this shape:
{
  "subject": "Economics (9750)",
  "subtopics": [
    { "code": "1.1", "title": "Functions" }
  ]
}

Syllabus text:
`;



function buildQuery(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      search.set(key, value);
    }
  });
  return search.toString();
}

function normalizeCategory(value) {
  if (!value) return value;
  if (value === "A-Level" || value === 'GCE "A" Levels') {
    return CANONICAL_CATEGORY;
  }
  return value;
}

export default function App() {
  const [subjects, setSubjects] = useState([]);
  const [subject, setSubject] = useState("All");
  const [subtopics, setSubtopics] = useState([]);
  const [subtopicSearch, setSubtopicSearch] = useState("");
  const [selectedSubtopic, setSelectedSubtopic] = useState(null);

  const [categories, setCategories] = useState([]);
  const [years, setYears] = useState([]);
  const [category, setCategory] = useState("All");
  const [year, setYear] = useState("All");
  const [document_name, setDocumentName] = useState("Document Name");
  const [questionType, setQuestionType] = useState(questionTypeOptions[0]);
  const [syllabusFile, setSyllabusFile] = useState(null);

  const [questions, setQuestions] = useState([]);
  const [status, setStatus] = useState({ type: "idle", message: "" });
  const [isLoading, setIsLoading] = useState(false);

  const buttonBase =
    "inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60";
  const primaryButton = `${buttonBase} bg-emerald-600 text-white hover:bg-emerald-700 focus-visible:ring-emerald-500`;
  const secondaryButton = `${buttonBase} border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 focus-visible:ring-slate-400`;
  const accentButton = `${buttonBase} bg-rose-600 text-white hover:bg-rose-700 focus-visible:ring-rose-500`;
  const statusStyles = {
    idle: "border-slate-200 bg-white text-slate-600",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-rose-200 bg-rose-50 text-rose-700",
  };

  const loadSubjects = async () => {
    try {
      const res = await fetch(`${API_URL}/subjects`);
      const data = await res.json();
      if (Array.isArray(data.subjects) && data.subjects.length > 0) {
        setSubjects(data.subjects);
        return;
      }
    } catch (err) {
      setStatus({ type: "error", message: "Could not load subjects from API." });
    }
    setSubjects(defaultSubjects);
  };

  const loadSubtopics = async (targetSubject = subject) => {
    if (!targetSubject || targetSubject === "All") {
      setSubtopics([]);
      setSelectedSubtopic(null);
      return;
    }
    try {
      const res = await fetch(`${API_URL}/subtopics?${buildQuery({ subject: targetSubject })}`);
      const data = await res.json();
      setSubtopics(Array.isArray(data.subtopics) ? data.subtopics : []);
      setSelectedSubtopic(null);
    } catch (err) {
      setStatus({ type: "error", message: "Could not load subtopics." });
    }
  };

  useEffect(() => {
    const loadFilters = async () => {
      try {
        const res = await fetch(`${API_URL}/questions/filters`);
        const data = await res.json();
        if (Array.isArray(data.categories) && data.categories.length > 0) {
          setCategories(data.categories);
        } else {
          setCategories(defaultCategories);
        }
        if (Array.isArray(data.years) && data.years.length > 0) {
          setYears(data.years);
        } else {
          setYears(fallbackYears);
        }
      } catch (err) {
        setStatus({ type: "error", message: "Could not load filters from API." });
      }
    };

    loadFilters();
    loadSubjects();
  }, []);

  useEffect(() => {
    loadSubtopics();
  }, [subject]);

  useEffect(() => {
    if (subject !== "All") {
      loadQuestions();
    }
  }, [subject, selectedSubtopic]);
  const filteredSubtopics = useMemo(() => {
    const term = subtopicSearch.trim().toLowerCase();
    if (!term) return subtopics;
    return subtopics.filter((subtopic) =>
      `${subtopic.code} ${subtopic.title}`.toLowerCase().includes(term)
    );
  }, [subtopics, subtopicSearch]);

  const syncContext = async (overrides = {}) => {
    const subjectLabel =
      overrides.subject_label ??
      overrides.subject ??
      (subject === "All" ? "Economics" : subject);
    const scraperConfig = {
      category: normalizeCategory(
        overrides.category ?? (category === "All" ? CANONICAL_CATEGORY : category)
      ),
      subject: overrides.scraper_subject ?? overrides.subject ?? (subject === "All" ? "Economics" : subject),
      year:
        overrides.year ??
        (year === "All" ? null : Number.isNaN(Number(year)) ? null : Number(year)),
      document_type: "Exam Papers",
      pages: 3,
      subject_label: subjectLabel,
    };
    await fetch(`${API_URL}/scraper/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scraperConfig),
    });
    const payload = {
      year: overrides.year ?? (Number(year) || new Date().getFullYear()),
      subject: overrides.subject ?? (subject === "All" ? "Economics" : subject),
      category: normalizeCategory(
        overrides.category ?? (category === "All" ? CANONICAL_CATEGORY : category)
      ),
      question_type: overrides.question_type ?? questionType,
      source_link: overrides.source_link ?? "",
      document_name: overrides.document_name ?? document_name,
    };
    await fetch(`${API_URL}/context`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  };

  const parseAiJson = (output) => {
    const cleaned = output.replace(/```json|```/g, "").trim();
    return JSON.parse(cleaned);
  };

  const loadQuestions = async () => {
    setIsLoading(true);
    setStatus({ type: "idle", message: "" });
    try {
      const params = {
        subject,
        year,
        category,
        subtopic: selectedSubtopic?.code,
      };
      const res = await fetch(`${API_URL}/questions?${buildQuery(params)}`);
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Could not fetch questions.");
      }
      const data = await res.json();
      setQuestions(Array.isArray(data.questions) ? data.questions : []);
      setStatus({ type: "success", message: `Loaded ${data.questions?.length || 0} questions.` });
    } catch (err) {
      setStatus({ type: "error", message: "Could not fetch questions." });
    } finally {
      setIsLoading(false);
    }
  };

  const extractSubtopicsFromSyllabus = async () => {
    if (!syllabusFile) {
      setStatus({ type: "error", message: "Select a syllabus PDF first." });
      return;
    }
    if (!window.puter?.ai?.chat) {
      setStatus({ type: "error", message: "Puter.js is not available." });
      return;
    }

    setIsLoading(true);
    setStatus({ type: "idle", message: "" });

    try {
      const formData = new FormData();
      formData.append("subject", subject);
      formData.append("file", syllabusFile);

      const uploadRes = await fetch(`${API_URL}/syllabus/extract`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const detail = await uploadRes.text();
        throw new Error(detail || "Syllabus extraction failed.");
      }

      const { text } = await uploadRes.json();
      const prompt = `${subtopicPrompt}${text}`;
      const aiResponse = await window.puter.ai.chat(prompt, { model: MODEL });
      const output =
        typeof aiResponse === "string"
          ? aiResponse
          : aiResponse?.message?.content || aiResponse?.content || JSON.stringify(aiResponse);

      const resultObj = parseAiJson(output);
      const extractedSubject =
        typeof resultObj.subject === "string" && resultObj.subject.trim().length > 0
          ? resultObj.subject.trim()
          : subject;
      const payload = {
        subject: extractedSubject === "All" ? "Economics" : extractedSubject,
        subtopics: resultObj.subtopics || [],
      };

      const subtopicRes = await fetch(`${API_URL}/subtopics/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const subtopicData = await subtopicRes.json();
      setStatus({
        type: "success",
        message: `Seeded ${subtopicData.created || 0} subtopics for ${payload.subject}.`,
      });
      await loadSubjects();
      setSubject(payload.subject);
      await loadSubtopics(payload.subject);
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Subtopic extraction failed." });
    } finally {
      setIsLoading(false);
    }
  };

  const runAiPipeline = async () => {
    if (!window.puter?.ai?.chat) {
      setStatus({ type: "error", message: "Puter.js is not available." });
      return;
    }

    setIsLoading(true);
    setStatus({ type: "idle", message: "" });

    try {
      await syncContext();
      const dataRes = await fetch(`${API_URL}/data`);
      if (!dataRes.ok) {
        const detail = await dataRes.text();
        throw new Error(detail || "Could not load document text.");
      }
      const dataPayload = await dataRes.json();
      const contextPayload = dataPayload.context;
      if (contextPayload?.document_name) {
        setDocumentName(contextPayload.document_name);
      }
      if (contextPayload?.subject && subject === "All") {
        setSubject(contextPayload.subject);
      }
      if (contextPayload?.category && category === "All") {
        setCategory(contextPayload.category);
      }
      if (typeof contextPayload?.year === "number" && year === "All") {
        setYear(contextPayload.year > 0 ? String(contextPayload.year) : "All");
      }
      if (contextPayload) {
        await syncContext(contextPayload);
      }
      const subjectForPrompt = subject === "All" ? "" : subject;
      const subtopicRes = subjectForPrompt
        ? await fetch(`${API_URL}/subtopics?${buildQuery({ subject: subjectForPrompt })}`)
        : null;
      const subtopicData = subtopicRes ? await subtopicRes.json() : { subtopics: [] };
      const subtopicLines = (subtopicData.subtopics || [])
        .map((item) => `${item.code} ${item.title}`)
        .join("\n");

      const { text } = dataPayload;
      const prompt = `${basePrompt}\nSUBTOPICS (use exact match):\n${subtopicLines}\n\n${text}`;

      const aiResponse = await window.puter.ai.chat(prompt, { model: MODEL });
      const output =
        typeof aiResponse === "string"
          ? aiResponse
          : aiResponse?.message?.content || aiResponse?.content || JSON.stringify(aiResponse);

      const resultObj = parseAiJson(output);

      const postRes = await fetch(`${API_URL}/ai-result`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ result: resultObj, context: contextPayload }),
      });
      const postData = await postRes.json();

      setStatus({ type: "success", message: `AI pipeline complete: ${postData.status || "ok"}` });
      await loadQuestions();
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Pipeline failed." });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-amber-100/40 to-slate-100 font-display text-slate-900">
      <div className="relative">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-24 left-24 h-72 w-72 rounded-full bg-amber-200/60 blur-3xl"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute bottom-10 right-8 h-64 w-64 rounded-full bg-sky-200/60 blur-3xl"
        />
      </div>
      <div className="relative z-10 mx-auto w-full max-w-7xl px-4 py-6 lg:px-6 lg:py-8">
        <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="order-2 flex flex-col gap-6 rounded-2xl border border-slate-200 bg-slate-100/80 p-5 shadow-sm lg:order-1 lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)]">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500 text-lg font-semibold text-white">
                G
              </div>
              <div>
                <h1 className="text-lg font-semibold text-slate-900">Grail Questions</h1>
                <p className="text-sm text-slate-600">Filter by syllabus chapter and year.</p>
              </div>
            </div>

            <div className="space-y-2 text-sm">
              <label htmlFor="subject" className="font-semibold text-slate-800">
                Subject
              </label>
              <select
                id="subject"
                value={subject}
                onChange={(event) => setSubject(event.target.value)}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
              >
                <option value="All">All subjects</option>
                {subjects.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-800">Subtopics</h2>
                <label htmlFor="subtopic-search" className="sr-only">
                  Search subtopics
                </label>
                <input
                  id="subtopic-search"
                  placeholder="Search subtopics"
                  value={subtopicSearch}
                  onChange={(event) => setSubtopicSearch(event.target.value)}
                  className="mt-2 w-full rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                />
              </div>
            </div>

            <div className="flex-1 space-y-2 overflow-auto pr-1">
              {filteredSubtopics.map((subtopic) => {
                const isActive = selectedSubtopic?.id === subtopic.id;
                return (
                  <button
                    type="button"
                    key={subtopic.id}
                    className={`w-full rounded-xl border px-3 py-2 text-left text-sm transition ${
                      isActive
                        ? "border-amber-300 bg-amber-50 shadow-sm"
                        : "border-transparent bg-white hover:border-slate-200"
                    }`}
                    onClick={() => setSelectedSubtopic(subtopic)}
                  >
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
                      {subtopic.code}
                    </span>
                    <span className="mt-1 block text-sm font-medium text-slate-900">
                      {subtopic.title}
                    </span>
                  </button>
                );
              })}
              {filteredSubtopics.length === 0 && (
                <p className="text-sm text-slate-500">No subtopics yet. Add them via the API.</p>
              )}
            </div>
          </aside>

          <main className="order-1 flex flex-col gap-6 lg:order-2">
            <section className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm backdrop-blur">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-amber-600">
                    Curate the question bank
                  </p>
                  <h2 className="mt-2 text-3xl font-semibold text-slate-900">
                    Curate the question bank
                  </h2>
                  <p className="mt-2 text-sm text-slate-600">
                    Select a context, sync it to the backend, then run extraction or load
                    matching questions.
                  </p>
                </div>
                <div className="flex items-start lg:justify-end">
                  <span
                    role="status"
                    className={`inline-flex items-center rounded-full border px-4 py-2 text-xs font-semibold ${
                      statusStyles[status.type] || statusStyles.idle
                    }`}
                  >
                    {status.message || "Ready"}
                  </span>
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">
                  Seed subtopics from syllabus
                </h3>
                <p className="mt-2 text-sm text-slate-600">
                  Upload the syllabus PDF for the selected subject and let Gemini extract the
                  subtopic list.
                </p>
              </div>
              <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center">
                <div className="flex-1 space-y-2">
                  <label htmlFor="syllabus-file" className="text-sm font-medium text-slate-700">
                    Syllabus PDF
                  </label>
                  <input
                    id="syllabus-file"
                    type="file"
                    accept="application/pdf"
                    onChange={(event) => setSyllabusFile(event.target.files?.[0] || null)}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm file:mr-4 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1 file:text-sm file:font-semibold file:text-slate-700 hover:file:bg-slate-200 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  />
                </div>
                <button
                  className={primaryButton}
                  onClick={extractSubtopicsFromSyllabus}
                  disabled={isLoading}
                >
                  Extract subtopics
                </button>
              </div>
              {syllabusFile && (
                <p className="mt-3 text-xs text-slate-500">Selected: {syllabusFile.name}</p>
              )}
            </section>

            <section className="grid gap-4 rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm md:grid-cols-3">
              <div className="space-y-2 text-sm">
                <label htmlFor="year" className="font-semibold text-slate-800">
                  Year
                </label>
                <select
                  id="year"
                  value={year}
                  onChange={(event) => setYear(event.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                >
                  <option value="All">All years</option>
                  {years.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2 text-sm">
                <label htmlFor="category" className="font-semibold text-slate-800">
                  Category
                </label>
                <select
                  id="category"
                  value={category}
                  onChange={(event) => setCategory(event.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                >
                  <option value="All">All categories</option>
                  {categories.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2 text-sm">
                <label htmlFor="questionType" className="font-semibold text-slate-800">
                  Question type
                </label>
                <select
                  id="questionType"
                  value={questionType}
                  onChange={(event) => setQuestionType(event.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                >
                  {questionTypeOptions.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
            </section>

            <section className="flex flex-wrap gap-3">
              <button
                type="button"
                className={secondaryButton}
                onClick={syncContext}
                disabled={isLoading}
              >
                Sync context
              </button>
              <button
                type="button"
                className={primaryButton}
                onClick={loadQuestions}
                disabled={isLoading}
              >
                {isLoading ? "Loading..." : "Load questions"}
              </button>
              <button
                type="button"
                className={accentButton}
                onClick={runAiPipeline}
                disabled={isLoading}
              >
                Run AI extraction
              </button>
            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Results</h3>
                <p className="text-xs uppercase tracking-wide text-slate-500">
                  {questions.length} total
                </p>
              </div>
              <div className="space-y-4">
                {questions.map((question) => (
                  <article
                    key={question.id}
                    className="grid gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:grid-cols-[140px_minmax(0,1fr)_90px]"
                  >
                    <div className="space-y-2 text-xs uppercase tracking-wide text-slate-500">
                      <span className="block">{question.year}</span>
                      <span className="block">{question.category}</span>
                      <span className="block">{question.question_type}</span>
                    </div>
                    <div>
                      <h3 className="text-base font-semibold text-slate-900">
                        {question.document_name || "Untitled document"}
                      </h3>
                      <p className="mt-2 text-sm text-slate-700">{question.question_text}</p>
                      <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                        <span className="rounded-full bg-slate-100 px-2 py-1">
                          {question.subject}
                        </span>
                        <span className="rounded-full bg-slate-100 px-2 py-1">
                          {question.source_link || "Source pending"}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-start lg:justify-center">
                      <div className="rounded-2xl bg-amber-50 px-4 py-3 text-center">
                        <p className="text-lg font-semibold text-slate-900">
                          {question.marks ?? "-"}
                        </p>
                        <p className="text-[10px] uppercase tracking-wide text-slate-500">
                          marks
                        </p>
                      </div>
                    </div>
                  </article>
                ))}
                {questions.length === 0 && (
                  <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-white/80 px-6 py-10 text-center">
                    <p className="text-sm font-semibold text-slate-800">
                      No questions yet. Pull a context or run extraction to populate results.
                    </p>
                    <p className="text-xs text-slate-500">
                      Sync the context, load questions, or run the AI extraction pipeline.
                    </p>
                    <button
                      type="button"
                      className={accentButton}
                      onClick={runAiPipeline}
                      disabled={isLoading}
                    >
                      Run AI extraction
                    </button>
                  </div>
                )}
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}
