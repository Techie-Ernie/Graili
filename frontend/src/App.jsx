import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MODEL = "gemini-2.5-flash-lite";

const defaultSubjects = [];
const defaultCategories = [];
const questionTypeOptions = ["exam", "understanding"];
const CANONICAL_CATEGORY = "GCE 'A' Levels";
const RESULTS_PAGE_SIZE = 8;
const DEFAULT_SCRAPER_PAGES = 2;
const MAX_SCRAPER_PAGES = 20;
const CLIENT_SESSION_STORAGE_KEY = "graili_client_session_id";


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
You are a strict information extraction engine. Extract syllabus subject, main themes, and subthemes from the input text.

Rules:
- A subject is the name of the subject and its subject code, e.g. Economics (Syllabus 9750), Physics (Syllabus 6091).
- Capture BOTH:
  1) Main themes (top-level code, e.g. "1", "2", "3")
  2) Subthemes (nested code, e.g. "1.1", "1.2.3")
- A theme/subtheme is a numeric code followed by a title.
- If a line is only a numeric code, use the next non-empty line as its title.
- Keep the most specific title associated with each code.
- Ignore non-theme content (prefaces, assessment objectives, admin sections).
- Do not invent or paraphrase.
- No duplicates by code.
- Return all items in a single flat list in ascending code order.

Output JSON ONLY in this shape:
{
  "subject": "Economics (9750)",
  "subtopics": [
    { "code": "1", "title": "Price mechanism and its applications" },
    { "code": "1.1", "title": "Demand and supply analysis and its applications" }
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

function compareCodes(a, b) {
  const aParts = String(a)
    .split(".")
    .map((part) => Number.parseInt(part, 10));
  const bParts = String(b)
    .split(".")
    .map((part) => Number.parseInt(part, 10));
  const maxLen = Math.max(aParts.length, bParts.length);
  for (let i = 0; i < maxLen; i += 1) {
    const aVal = Number.isNaN(aParts[i]) ? -1 : (aParts[i] ?? -1);
    const bVal = Number.isNaN(bParts[i]) ? -1 : (bParts[i] ?? -1);
    if (aVal !== bVal) return aVal - bVal;
  }
  return String(a).localeCompare(String(b));
}

function sanitizePageCount(value) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed) || parsed < 1) return 1;
  return Math.min(parsed, MAX_SCRAPER_PAGES);
}

function getCurrentPage() {
  if (typeof window === "undefined") return "user";
  const hashPath = (window.location.hash || "").replace(/^#/, "");
  const path = hashPath || window.location.pathname || "/";
  return path === "/admin" ? "admin" : "user";
}

function getOrCreateClientSessionId() {
  if (typeof window === "undefined") return "server-session";
  const existing = window.sessionStorage.getItem(CLIENT_SESSION_STORAGE_KEY);
  if (existing) return existing;
  const generated =
    window.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.sessionStorage.setItem(CLIENT_SESSION_STORAGE_KEY, generated);
  return generated;
}

export default function App() {
  const [currentPage, setCurrentPage] = useState(() => getCurrentPage());
  const [subjects, setSubjects] = useState([]);
  const [subject, setSubject] = useState("All");
  const [subtopics, setSubtopics] = useState([]);
  const [selectedSubtopicCodes, setSelectedSubtopicCodes] = useState([]);
  const [expandedThemeCodes, setExpandedThemeCodes] = useState([]);
  const [collections, setCollections] = useState([]);
  const [selectedCollectionIds, setSelectedCollectionIds] = useState([]);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [collectionForAddId, setCollectionForAddId] = useState("");

  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("All");
  const [document_name, setDocumentName] = useState("Document Name");
  const [questionType, setQuestionType] = useState(questionTypeOptions[0]);
  const [questionSearch, setQuestionSearch] = useState("");
  const [debouncedQuestionSearch, setDebouncedQuestionSearch] = useState("");
  const [scraperPages, setScraperPages] = useState(DEFAULT_SCRAPER_PAGES);
  const [syllabusFile, setSyllabusFile] = useState(null);
  const [questionUploadFiles, setQuestionUploadFiles] = useState([]);

  const [scrapedQuestions, setScrapedQuestions] = useState([]);
  const [uploadedQuestions, setUploadedQuestions] = useState([]);
  const [resultsView, setResultsView] = useState("scraped");
  const [resultsPage, setResultsPage] = useState(1);
  const [sourceCounts, setSourceCounts] = useState({ scraped: 0, uploaded: 0 });
  const [status, setStatus] = useState({ type: "idle", message: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [isScrapingDocs, setIsScrapingDocs] = useState(false);
  const [extractionProgress, setExtractionProgress] = useState(null);
  const [clientSessionId] = useState(() => getOrCreateClientSessionId());
  const isAdminPage = currentPage === "admin";

  const apiFetch = (url, options = {}) => {
    const headers = new Headers(options.headers || {});
    if (clientSessionId && !headers.has("X-Client-Session")) {
      headers.set("X-Client-Session", clientSessionId);
    }
    return fetch(url, { ...options, headers });
  };

  const buttonBase =
    "inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60";
  const primaryButton = `${buttonBase} bg-emerald-600 text-white hover:bg-emerald-700 focus-visible:ring-emerald-500`;
  const accentButton = `${buttonBase} bg-rose-600 text-white hover:bg-rose-700 focus-visible:ring-rose-500`;
  const statusStyles = {
    idle: "border-slate-200 bg-white text-slate-600",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-rose-200 bg-rose-50 text-rose-700",
  };
  const totalQuestions = scrapedQuestions.length + uploadedQuestions.length;
  const displayedQuestions = resultsView === "uploaded" ? uploadedQuestions : scrapedQuestions;
  const displayedSourceLabel = resultsView === "uploaded" ? "uploaded" : "scraped";
  const displayedEmptyLabel = resultsView === "uploaded"
    ? "No user-uploaded questions for the current filters."
    : "No scraped questions for the current filters.";
  const totalResultPages = Math.max(1, Math.ceil(displayedQuestions.length / RESULTS_PAGE_SIZE));
  const safeResultsPage = Math.min(resultsPage, totalResultPages);
  const pagedQuestions = displayedQuestions.slice(
    (safeResultsPage - 1) * RESULTS_PAGE_SIZE,
    safeResultsPage * RESULTS_PAGE_SIZE
  );

  const loadSubjects = async () => {
    try {
      const res = await apiFetch(`${API_URL}/subjects`);
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
      setSelectedSubtopicCodes([]);
      return;
    }
    try {
      const res = await apiFetch(`${API_URL}/subtopics?${buildQuery({ subject: targetSubject })}`);
      const data = await res.json();
      setSubtopics(Array.isArray(data.subtopics) ? data.subtopics : []);
      setSelectedSubtopicCodes([]);
    } catch (err) {
      setStatus({ type: "error", message: "Could not load subtopics." });
    }
  };

  const loadCollections = async (targetSubject = subject) => {
    try {
      const query = targetSubject && targetSubject !== "All"
        ? buildQuery({ subject: targetSubject })
        : "";
      const res = await apiFetch(`${API_URL}/collections${query ? `?${query}` : ""}`);
      if (!res.ok) {
        throw new Error("Could not load collections.");
      }
      const data = await res.json();
      const rows = Array.isArray(data.collections) ? data.collections : [];
      setCollections(rows);

      const validIds = new Set(rows.map((row) => Number(row.id)));
      setSelectedCollectionIds((previous) =>
        previous.filter((id) => validIds.has(Number(id)))
      );
      setCollectionForAddId((previous) => {
        if (previous && validIds.has(Number(previous))) return previous;
        return rows[0] ? String(rows[0].id) : "";
      });
    } catch (err) {
      setStatus({ type: "error", message: "Could not load collections." });
    }
  };

  const loadFilters = async () => {
    try {
      const res = await apiFetch(`${API_URL}/questions/filters`);
      const data = await res.json();
      if (Array.isArray(data.categories) && data.categories.length > 0) {
        setCategories(data.categories);
      } else {
        setCategories(defaultCategories);
      }
      if (data.source_counts && typeof data.source_counts === "object") {
        setSourceCounts({
          scraped: Number(data.source_counts.scraped || 0),
          uploaded: Number(data.source_counts.uploaded || 0),
        });
      }
    } catch (err) {
      setStatus({ type: "error", message: "Could not load filters from API." });
    }
  };

  useEffect(() => {
    loadFilters();
    loadSubjects();
  }, []);

  useEffect(() => {
    const handleRouteChange = () => setCurrentPage(getCurrentPage());
    window.addEventListener("hashchange", handleRouteChange);
    window.addEventListener("popstate", handleRouteChange);
    return () => {
      window.removeEventListener("hashchange", handleRouteChange);
      window.removeEventListener("popstate", handleRouteChange);
    };
  }, []);

  useEffect(() => {
    loadSubtopics();
    loadCollections();
  }, [subject]);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setDebouncedQuestionSearch(questionSearch.trim());
    }, 250);
    return () => window.clearTimeout(handle);
  }, [questionSearch]);

  useEffect(() => {
    if (subject === "All") {
      setScrapedQuestions([]);
      setUploadedQuestions([]);
      setResultsPage(1);
      return;
    }
    loadQuestions();
  }, [
    subject,
    category,
    questionType,
    debouncedQuestionSearch,
    selectedSubtopicCodes.join("|"),
    selectedCollectionIds.join("|"),
  ]);

  useEffect(() => {
    setResultsPage(1);
  }, [resultsView, scrapedQuestions.length, uploadedQuestions.length]);

  const themeGroups = useMemo(() => {
    const groups = new Map();
    const normalized = (subtopics || [])
      .filter((item) => item?.code && item?.title)
      .map((item) => ({ ...item, code: String(item.code).trim(), title: String(item.title).trim() }));

    normalized.forEach((item) => {
      if (!item.code || !item.title) return;
      const isMainTheme = !item.code.includes(".");
      const mainCode = isMainTheme ? item.code : item.code.split(".")[0];
      if (!groups.has(mainCode)) {
        groups.set(mainCode, { code: mainCode, title: "", subthemes: [] });
      }
      const group = groups.get(mainCode);
      if (isMainTheme) {
        group.title = item.title;
      } else {
        group.subthemes.push(item);
      }
    });

    const finalGroups = Array.from(groups.values())
      .map((group) => {
        const sortedSubthemes = group.subthemes.sort((a, b) => compareCodes(a.code, b.code));
        const fallbackTitle = sortedSubthemes[0]?.title
          ? `Theme ${group.code}: ${sortedSubthemes[0].title}`
          : `Theme ${group.code}`;
        return {
          ...group,
          title: group.title || fallbackTitle,
          subthemes: sortedSubthemes,
        };
      })
      .sort((a, b) => compareCodes(a.code, b.code));

    return finalGroups;
  }, [subtopics]);

  const selectedCodeSet = useMemo(() => new Set(selectedSubtopicCodes), [selectedSubtopicCodes]);

  useEffect(() => {
    const validCodes = new Set(themeGroups.map((group) => group.code));
    setExpandedThemeCodes((previous) => previous.filter((code) => validCodes.has(code)));
  }, [themeGroups]);

  const toggleCodes = (codes, shouldSelect) => {
    setSelectedSubtopicCodes((previous) => {
      const next = new Set(previous);
      codes.forEach((code) => {
        if (shouldSelect) {
          next.add(code);
        } else {
          next.delete(code);
        }
      });
      return Array.from(next).sort(compareCodes);
    });
  };

  const toggleThemeExpanded = (themeCode) => {
    setExpandedThemeCodes((previous) => {
      if (previous.includes(themeCode)) {
        return previous.filter((code) => code !== themeCode);
      }
      return [...previous, themeCode];
    });
  };

  const clearSubtopicSelection = () => setSelectedSubtopicCodes([]);
  const selectedCollectionSet = useMemo(
    () => new Set(selectedCollectionIds.map((id) => String(id))),
    [selectedCollectionIds]
  );

  const toggleCollectionSelection = (collectionId, shouldSelect) => {
    const stringId = String(collectionId);
    setSelectedCollectionIds((previous) => {
      const next = new Set(previous.map((value) => String(value)));
      if (shouldSelect) {
        next.add(stringId);
      } else {
        next.delete(stringId);
      }
      return Array.from(next);
    });
  };

  const clearCollectionSelection = () => setSelectedCollectionIds([]);

  const createCollection = async () => {
    const name = newCollectionName.trim();
    if (!name) {
      setStatus({ type: "error", message: "Enter a collection name first." });
      return;
    }
    if (subject === "All") {
      setStatus({ type: "error", message: "Select a subject before creating a collection." });
      return;
    }
    try {
      const targetSubject = subject;
      const res = await apiFetch(`${API_URL}/collections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          subject: targetSubject,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Could not create collection.");
      }
      const data = await res.json();
      const createdCollection = data.collection;
      setNewCollectionName("");
      setStatus({
        type: "success",
        message: createdCollection?.created === false
          ? `Collection "${createdCollection.name}" already exists.`
          : `Created collection "${createdCollection?.name || name}".`,
      });
      await loadCollections(targetSubject);
      if (createdCollection?.id) {
        setCollectionForAddId(String(createdCollection.id));
      }
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Could not create collection." });
    }
  };

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
      year: overrides.year ?? null,
      document_type: "Exam Papers",
      pages: sanitizePageCount(overrides.pages ?? scraperPages),
      subject_label: subjectLabel,
    };
    await apiFetch(`${API_URL}/scraper/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scraperConfig),
    });
    const payload = {
      year: overrides.year ?? 0,
      subject: overrides.subject ?? (subject === "All" ? "Economics" : subject),
      category: normalizeCategory(
        overrides.category ?? (category === "All" ? CANONICAL_CATEGORY : category)
      ),
      question_type: overrides.question_type ?? questionType,
      source_link: overrides.source_link ?? "",
      document_name: overrides.document_name ?? document_name,
      source_type: overrides.source_type ?? "scraped",
      client_session_id: clientSessionId,
    };
    await apiFetch(`${API_URL}/context`, {
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
    if (subject === "All") {
      setScrapedQuestions([]);
      setUploadedQuestions([]);
      return;
    }
    setIsLoading(true);
    setStatus({ type: "idle", message: "" });
    try {
      const params = {
        subject,
        category,
        question_type: questionType,
        search: debouncedQuestionSearch,
        subtopics: selectedSubtopicCodes.join(","),
        collections: selectedCollectionIds.join(","),
      };
      const res = await apiFetch(`${API_URL}/questions?${buildQuery(params)}`);
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Could not fetch questions.");
      }
      const data = await res.json();
      const scraped = Array.isArray(data.scraped_questions) ? data.scraped_questions : [];
      const uploaded = Array.isArray(data.uploaded_questions) ? data.uploaded_questions : [];
      setScrapedQuestions(scraped);
      setUploadedQuestions(uploaded);
      setStatus({ type: "success", message: `Loaded ${scraped.length + uploaded.length} questions.` });
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

      const uploadRes = await apiFetch(`${API_URL}/syllabus/extract`, {
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
      const cleanedSubtopics = Array.isArray(resultObj.subtopics)
        ? resultObj.subtopics
            .filter((item) => item?.code && item?.title)
            .map((item) => ({
              code: String(item.code).trim(),
              title: String(item.title).trim(),
            }))
            .filter((item) => item.code && item.title)
            .sort((a, b) => compareCodes(a.code, b.code))
        : [];
      const payload = {
        subject: extractedSubject === "All" ? "Economics" : extractedSubject,
        subtopics: cleanedSubtopics,
      };

      const subtopicRes = await apiFetch(`${API_URL}/subtopics/bulk`, {
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

  const pushContext = async (contextPayload) => {
    await apiFetch(`${API_URL}/context`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(contextPayload),
    });
  };

  const processDocumentPayloads = async (documentPayloads, label) => {
    const total = documentPayloads.length;
    setExtractionProgress({
      label,
      current: 0,
      total,
      document_name: "",
    });
    try {
      for (let index = 0; index < documentPayloads.length; index += 1) {
        const { text, context } = documentPayloads[index] || {};
        const contextPayload = {
          ...(context || {}),
          client_session_id: clientSessionId,
        };

        setExtractionProgress((previous) => ({
          label,
          current: index + 1,
          total,
          document_name: contextPayload?.document_name || previous?.document_name || "",
        }));

        if (contextPayload?.document_name) {
          setDocumentName(contextPayload.document_name);
        }
        if (contextPayload?.subject && subject === "All") {
          setSubject(contextPayload.subject);
        }
        if (contextPayload?.category && category === "All") {
          setCategory(contextPayload.category);
        }
        if (contextPayload) {
          await pushContext(contextPayload);
        }

        const subjectForPrompt = contextPayload?.subject || (subject === "All" ? "" : subject);
        const subtopicRes = subjectForPrompt
          ? await apiFetch(`${API_URL}/subtopics?${buildQuery({ subject: subjectForPrompt })}`)
          : null;
        const subtopicData = subtopicRes ? await subtopicRes.json() : { subtopics: [] };
        const subtopicLines = (subtopicData.subtopics || [])
          .slice()
          .sort((a, b) => compareCodes(a.code, b.code))
          .map((item) => `${item.code} ${item.title}`)
          .join("\n");

        const prompt = `${basePrompt}\nSUBTOPICS (use exact match):\n${subtopicLines}\n\n${text}`;

        setStatus({
          type: "idle",
          message: `Running ${label} extraction ${index + 1} of ${documentPayloads.length}...`,
        });

        const aiResponse = await window.puter.ai.chat(prompt, { model: MODEL });
        const output =
          typeof aiResponse === "string"
            ? aiResponse
            : aiResponse?.message?.content || aiResponse?.content || JSON.stringify(aiResponse);

        const resultObj = parseAiJson(output);

        const postRes = await apiFetch(`${API_URL}/ai-result`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ result: resultObj, context: contextPayload }),
        });
        if (!postRes.ok) {
          const detail = await postRes.text();
          throw new Error(detail || "Could not save AI result.");
        }
        await postRes.json();
      }
    } finally {
      setExtractionProgress(null);
    }
  };

  const runScrapedAiPipeline = async () => {
    if (!window.puter?.ai?.chat) {
      setStatus({ type: "error", message: "Puter.js is not available." });
      return;
    }
    if (!subject || subject === "All") {
      setStatus({ type: "error", message: "Select a subject before scraping documents." });
      return;
    }

    setIsLoading(true);
    setStatus({ type: "idle", message: "" });

    try {
      await syncContext({ source_type: "scraped" });
      setIsScrapingDocs(true);
      setStatus({ type: "idle", message: "Scraping and downloading documents..." });
      const dataRes = await apiFetch(`${API_URL}/data`);
      if (!dataRes.ok) {
        const detail = await dataRes.text();
        throw new Error(detail || "Could not load scraped document text.");
      }
      const dataPayload = await dataRes.json();
      setIsScrapingDocs(false);
      const documentPayloads = Array.isArray(dataPayload.documents) && dataPayload.documents.length > 0
        ? dataPayload.documents
        : [{ text: dataPayload.text, context: dataPayload.context }];

      await processDocumentPayloads(documentPayloads, "scraped");
      setStatus({
        type: "success",
        message: `Scraped extraction complete: processed ${documentPayloads.length} document(s).`,
      });
      await loadQuestions();
      await loadFilters();
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Scraped pipeline failed." });
    } finally {
      setIsScrapingDocs(false);
      setIsLoading(false);
    }
  };

  const runUploadedAiPipeline = async () => {
    if (!window.puter?.ai?.chat) {
      setStatus({ type: "error", message: "Puter.js is not available." });
      return;
    }
    if (questionUploadFiles.length === 0) {
      setStatus({ type: "error", message: "Select at least one question PDF to upload." });
      return;
    }

    setIsLoading(true);
    setStatus({ type: "idle", message: "" });

    try {
      const formData = new FormData();
      formData.append("subject", subject === "All" ? "Economics" : subject);
      formData.append("category", category === "All" ? CANONICAL_CATEGORY : category);
      formData.append("client_session_id", clientSessionId);
      questionUploadFiles.forEach((file) => {
        formData.append("files", file);
      });

      const uploadRes = await apiFetch(`${API_URL}/uploads/question-documents/extract`, {
        method: "POST",
        body: formData,
      });
      if (!uploadRes.ok) {
        const detail = await uploadRes.text();
        throw new Error(detail || "Could not extract uploaded document text.");
      }

      const uploadPayload = await uploadRes.json();
      const documentPayloads = Array.isArray(uploadPayload.documents) && uploadPayload.documents.length > 0
        ? uploadPayload.documents
        : [{ text: uploadPayload.text, context: uploadPayload.context }];

      await processDocumentPayloads(documentPayloads, "uploaded");
      setStatus({
        type: "success",
        message: `Uploaded extraction complete: processed ${documentPayloads.length} document(s).`,
      });
      setQuestionUploadFiles([]);
      await loadQuestions();
      await loadFilters();
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Uploaded pipeline failed." });
    } finally {
      setIsLoading(false);
    }
  };

  const addDocumentToCollection = async (question) => {
    if (!collectionForAddId) {
      setStatus({ type: "error", message: "Select a collection before adding documents." });
      return;
    }
    if (!question?.document_name || !question?.subject || !question?.source_type) {
      setStatus({ type: "error", message: "Missing document details for collection add." });
      return;
    }
    try {
      const res = await apiFetch(`${API_URL}/collections/documents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_id: Number(collectionForAddId),
          subject: question.subject,
          source_type: question.source_type,
          document_name: question.document_name,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Could not add document to collection.");
      }
      const data = await res.json();
      setStatus({
        type: "success",
        message: data.added
          ? `Added "${question.document_name}" to collection.`
          : `"${question.document_name}" is already in this collection.`,
      });
      await loadCollections(subject);
      await loadQuestions();
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Could not add document to collection." });
    }
  };

  const renderQuestionCards = (questions, sourceLabel) =>
    questions.map((question) => (
      <article
        key={`${sourceLabel}-${question.id}`}
        className={`grid gap-3 rounded-xl border p-3 shadow-sm lg:grid-cols-[120px_minmax(0,1fr)] ${
          sourceLabel === "scraped"
            ? "border-amber-200 bg-amber-50/40"
            : "border-sky-200 bg-sky-50/40"
        }`}
      >
        <div className="space-y-2 text-xs uppercase tracking-wide text-slate-500">
          <span className="block">{question.category}</span>
          <span className="block">{question.question_type}</span>
        </div>
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            {question.document_name || "Untitled document"}
          </h3>
          <p className="mt-1 text-sm text-slate-700">{question.question_text}</p>
          <div className="mt-2 flex flex-wrap gap-1.5 text-xs text-slate-500">
            <span className="rounded-full bg-slate-100 px-2 py-1">{question.subject}</span>
            <span
              className={`rounded-full px-2 py-1 font-semibold ${
                sourceLabel === "scraped"
                  ? "bg-amber-200/60 text-amber-900"
                  : "bg-sky-200/60 text-sky-900"
              }`}
            >
              {sourceLabel === "scraped" ? "Scraped source" : "User upload"}
            </span>
            {question.source_link && (
              <a className="rounded-full bg-amber-100 px-2 py-1" href={question.source_link}>
                Source 🔗
              </a>
            )}
            {Array.isArray(question.collections) && question.collections.length > 0 ? (
              question.collections.map((collectionName) => (
                <span
                  key={`${question.id}-${collectionName}`}
                  className="rounded-full bg-indigo-100 px-2 py-1 font-semibold text-indigo-800"
                >
                  {collectionName}
                </span>
              ))
            ) : (
              <span className="rounded-full bg-slate-200 px-2 py-1">No collection</span>
            )}
            <button
              type="button"
              onClick={() => addDocumentToCollection(question)}
              className="rounded-full bg-emerald-100 px-2 py-1 font-semibold text-emerald-800 hover:bg-emerald-200 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={!collectionForAddId}
              title={collectionForAddId ? "Add this document to the selected collection" : "Select a collection first"}
            >
              Add To Collection
            </button>
          </div>
        </div>
      </article>
    ));

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
        <div className="mb-4 flex justify-end">
          <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 text-sm shadow-sm">
            <a
              href="/"
              className={`rounded-lg px-3 py-1.5 font-semibold ${
                !isAdminPage ? "bg-amber-100 text-amber-900" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              User
            </a>
            <a
              href="/#/admin"
              className={`rounded-lg px-3 py-1.5 font-semibold ${
                isAdminPage ? "bg-rose-100 text-rose-900" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              Admin
            </a>
          </div>
        </div>
        <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="order-2 flex flex-col gap-4 rounded-2xl border border-slate-200 bg-slate-100/80 p-5 shadow-sm lg:order-1 lg:sticky lg:top-4 lg:h-[calc(100vh-1.5rem)]">
            <div className="flex items-center gap-4">
              
              <div>
                <h1 className="text-2xl font-semibold text-slate-900">Graili</h1>
                <p className="text-sm text-slate-600">(Holy) Grail Improved</p>
                  <p className="text-sm text-slate-600 font-bold">By Techie Ernie</p>

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

            <div className="space-y-2 rounded-xl border border-slate-200 bg-white/70 p-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-800">Collections</h2>
                <button
                  type="button"
                  onClick={clearCollectionSelection}
                  className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  Clear
                </button>
              </div>
              <div className="flex gap-2">
                <input
                  value={newCollectionName}
                  onChange={(event) => setNewCollectionName(event.target.value)}
                  placeholder="New collection"
                  className="min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                />
                <button
                  type="button"
                  onClick={createCollection}
                  className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700"
                >
                  Add
                </button>
              </div>

              <div className="space-y-1">
                <label htmlFor="collection-add-target" className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Active Collection For Add
                </label>
                <select
                  id="collection-add-target"
                  value={collectionForAddId}
                  onChange={(event) => setCollectionForAddId(event.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                >
                  <option value="">Select collection</option>
                  {collections.map((collection) => (
                    <option key={collection.id} value={collection.id}>
                      {collection.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <p className="text-xs text-slate-500">{selectedCollectionIds.length} selected</p>
                <div className="max-h-28 space-y-1 overflow-auto pr-1">
                  {collections.map((collection) => (
                    <label key={collection.id} className="flex items-start gap-2 text-sm text-slate-700">
                      <input
                        type="checkbox"
                        checked={selectedCollectionSet.has(String(collection.id))}
                        onChange={(event) => toggleCollectionSelection(collection.id, event.target.checked)}
                        className="mt-0.5 h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      <span>
                        {collection.name}
                        <span className="ml-1 text-xs text-slate-500">({collection.documents_count || 0})</span>
                      </span>
                    </label>
                  ))}
                  {collections.length === 0 && (
                    <p className="text-xs text-slate-500">No collections yet.</p>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-800">Chapters</h2>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setExpandedThemeCodes(themeGroups.map((group) => group.code))}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                  >
                    Expand all
                  </button>
                  <button
                    type="button"
                    onClick={() => setExpandedThemeCodes([])}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                  >
                    Collapse all
                  </button>
                  <button
                    type="button"
                    onClick={clearSubtopicSelection}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <p className="text-xs text-slate-500">
                {selectedSubtopicCodes.length} selected
              </p>
            </div>

            <div className="flex-1 space-y-3 overflow-auto pr-1 min-h-[22rem]">
              {themeGroups.map((group) => {
                const themeCodes = [group.code, ...group.subthemes.map((item) => item.code)];
                const selectedCount = themeCodes.filter((code) => selectedCodeSet.has(code)).length;
                const isChecked = themeCodes.length > 0 && selectedCount === themeCodes.length;
                const isPartial = selectedCount > 0 && selectedCount < themeCodes.length;
                const isExpanded = expandedThemeCodes.includes(group.code);

                return (
                  <div key={group.code} className="rounded-xl border border-slate-200 bg-white p-3">
                    <div className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        ref={(element) => {
                          if (element) {
                            element.indeterminate = isPartial;
                          }
                        }}
                        onChange={(event) => toggleCodes(themeCodes, event.target.checked)}
                        className="mt-0.5 h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      <div className="min-w-0 flex-1">
                        <button
                          type="button"
                          onClick={() => toggleThemeExpanded(group.code)}
                          className="flex w-full items-center justify-between text-left text-sm font-semibold text-slate-800"
                        >
                          <span className="pr-2">{group.code} {group.title}</span>
                          <span className="shrink-0 text-xs text-slate-500">
                            {isExpanded ? "Hide" : "Show"} ({group.subthemes.length})
                          </span>
                        </button>

                        {isExpanded && (
                          <div className="mt-2 space-y-2 pl-2">
                            {group.subthemes.map((subtheme) => (
                              <label
                                key={subtheme.id ?? subtheme.code}
                                className="flex items-start gap-2 text-sm text-slate-700"
                              >
                                <input
                                  type="checkbox"
                                  checked={selectedCodeSet.has(subtheme.code)}
                                  onChange={(event) => toggleCodes([subtheme.code], event.target.checked)}
                                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                                />
                                <span>{subtheme.code} {subtheme.title}</span>
                              </label>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              {themeGroups.length === 0 && (
                <p className="text-sm text-slate-500">No themes yet. Extract them from the syllabus PDF.</p>
              )}
            </div>
          </aside>

          <main className="order-1 flex flex-col gap-6 lg:order-2">
            <section className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm backdrop-blur">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <h2 className="mt-2 text-3xl font-semibold text-slate-900">
                    {isAdminPage ? "Admin console" : "Curate the question bank"}
                  </h2>
                  <p className="mt-2 text-sm text-slate-600">
                    {isAdminPage
                      ? "Run manual chapter seeding and Grail scraping jobs."
                      : "Upload question documents, apply filters, and load matching results."}
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

            {isAdminPage && (
              <section className="rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">
                    Seed chapters from syllabus
                  </h3>
                  <p className="mt-2 text-sm text-slate-600">
                    Upload the syllabus PDF for the selected subject to extract subtopics.
                  </p>
                </div>
                <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-end">
                  <div className="flex-1 space-y-2">
                    <label htmlFor="syllabus-file" className="text-sm font-medium text-slate-700">
                      Syllabus PDF
                    </label>
                    <input
                      id="syllabus-file"
                      type="file"
                      accept="application/pdf"
                      onChange={(event) => setSyllabusFile(event.target.files?.[0] || null)}
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm file:mr-4 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-slate-700 hover:file:bg-slate-200 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                    />
                  </div>
                  <button
                    className={`${primaryButton} h-11`}
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
            )}

            {isAdminPage && (
              <section className="rounded-2xl border border-amber-200 bg-amber-50/40 p-6 shadow-sm">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">
                    Scrape Grail documents
                  </h3>
                  <p className="mt-2 text-sm text-slate-600">
                    Manually scrape Grail papers for the selected subject and extract questions.
                  </p>
                </div>
                <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-end">
                  <div className="w-full space-y-2 lg:max-w-xs">
                    <label htmlFor="scraperPages" className="text-sm font-medium text-slate-700">
                      Scraper pages
                    </label>
                    <input
                      id="scraperPages"
                      type="number"
                      min={1}
                      max={MAX_SCRAPER_PAGES}
                      step={1}
                      value={scraperPages}
                      onChange={(event) => setScraperPages(sanitizePageCount(event.target.value))}
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                    />
                    <p className="text-xs text-slate-500">
                      Number of result pages to scrape (1-{MAX_SCRAPER_PAGES}).
                    </p>
                  </div>
                  <button
                    type="button"
                    className={accentButton}
                    onClick={runScrapedAiPipeline}
                    disabled={isLoading || subject === "All"}
                    title={subject === "All" ? "Select a subject first" : "Run manual Grail scrape"}
                  >
                    Scrape Grail
                  </button>
                </div>
              </section>
            )}

            <section className="rounded-2xl border border-sky-200 bg-sky-50/40 p-6 shadow-sm">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">
                  Upload question documents
                </h3>
                <p className="mt-2 text-sm text-slate-600">
                  Upload your own question PDFs and extract them into the user-uploaded table.
                </p>
              </div>
              <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-end">
                <div className="flex-1 space-y-2">
                  <label htmlFor="question-upload-files" className="text-sm font-medium text-slate-700">
                    Question PDFs
                  </label>
                  <input
                    id="question-upload-files"
                    type="file"
                    accept="application/pdf"
                    multiple
                    onChange={(event) => setQuestionUploadFiles(Array.from(event.target.files || []))}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm file:mr-4 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-slate-700 hover:file:bg-slate-200 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  />
                </div>
                <button
                  className={`${accentButton} h-11`}
                  onClick={runUploadedAiPipeline}
                  disabled={isLoading}
                >
                  Extract uploaded docs
                </button>
              </div>
              {questionUploadFiles.length > 0 && (
                <p className="mt-3 text-xs text-slate-500">
                  Selected {questionUploadFiles.length} file(s).
                </p>
              )}
            </section>

            <section className="grid gap-4 rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-sm md:grid-cols-3">
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

              <div className="space-y-2 text-sm">
                <label htmlFor="questionSearch" className="font-semibold text-slate-800">
                  Search questions
                </label>
                <div className="flex gap-2">
                  <input
                    id="questionSearch"
                    value={questionSearch}
                    onChange={(event) => setQuestionSearch(event.target.value)}
                    placeholder="e.g. price controls, inflation"
                    className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  />
                  <button
                    type="button"
                    onClick={() => setQuestionSearch("")}
                    disabled={!questionSearch}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600 shadow-sm hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                    title="Clear search"
                  >
                    Clear
                  </button>
                </div>
                <p className="text-xs text-slate-500">
                  Filters by question text (space-separated terms).
                </p>
              </div>

            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Results</h3>
                <div className="flex gap-2 text-xs uppercase tracking-wide text-slate-500">
                  <span>{totalQuestions} loaded</span>
                  <span>DB scraped: {sourceCounts.scraped}</span>
                  <span>DB uploaded: {sourceCounts.uploaded}</span>
                </div>
              </div>
              <div className="space-y-4 rounded-2xl border border-slate-200 bg-white/70 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1">
                    <button
                      type="button"
                      onClick={() => setResultsView("scraped")}
                      className={`rounded-lg px-3 py-1.5 text-sm font-semibold ${
                        resultsView === "scraped"
                          ? "bg-amber-100 text-amber-900"
                          : "text-slate-600 hover:bg-slate-100"
                      }`}
                    >
                      Scraped ({scrapedQuestions.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setResultsView("uploaded")}
                      className={`rounded-lg px-3 py-1.5 text-sm font-semibold ${
                        resultsView === "uploaded"
                          ? "bg-sky-100 text-sky-900"
                          : "text-slate-600 hover:bg-slate-100"
                      }`}
                    >
                      Uploaded ({uploadedQuestions.length})
                    </button>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <button
                      type="button"
                      onClick={() => setResultsPage((previous) => Math.max(1, previous - 1))}
                      className="rounded-md border border-slate-200 bg-white px-2 py-1 disabled:opacity-50"
                      disabled={safeResultsPage <= 1}
                    >
                      Prev
                    </button>
                    <span>
                      Page {safeResultsPage} / {totalResultPages}
                    </span>
                    <button
                      type="button"
                      onClick={() => setResultsPage((previous) => Math.min(totalResultPages, previous + 1))}
                      className="rounded-md border border-slate-200 bg-white px-2 py-1 disabled:opacity-50"
                      disabled={safeResultsPage >= totalResultPages}
                    >
                      Next
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  {renderQuestionCards(pagedQuestions, displayedSourceLabel)}
                  {displayedQuestions.length === 0 && (
                    <p className="rounded-xl border border-dashed border-slate-300 bg-white px-4 py-3 text-sm text-slate-600">
                      {displayedEmptyLabel}
                    </p>
                  )}
                </div>
              </div>
            </section>
          </main>
        </div>
      </div>

      {isScrapingDocs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-emerald-600" />
              <div>
                <p className="text-sm font-semibold text-slate-900">Scraping documents...</p>
                <p className="mt-1 text-xs text-slate-600">
                  Fetching and downloading papers from Grail. This can take a while.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {!isScrapingDocs && extractionProgress && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="space-y-3">
              <div>
                <p className="text-sm font-semibold text-slate-900">
                  Running {extractionProgress.label} extraction...
                </p>
                <p className="mt-1 text-xs text-slate-600">
                  Processing document {extractionProgress.current} / {extractionProgress.total}
                  {extractionProgress.document_name ? `: ${extractionProgress.document_name}` : ""}
                </p>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-2 rounded-full bg-emerald-600 transition-[width]"
                  style={{
                    width: `${
                      extractionProgress.total > 0
                        ? Math.min(
                            100,
                            Math.round((extractionProgress.current / extractionProgress.total) * 100)
                          )
                        : 0
                    }%`,
                  }}
                />
              </div>
              <p className="text-xs text-slate-500">
                This runs one document at a time (AI call + save). Keep this tab open.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

}
