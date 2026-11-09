export interface ParsedTitle {
  title: string;
  year: string | null;
}

// "Matrix, The" or "Triplets of Belleville, The (Les triplettes de Belleville)"
const TRAILING_ARTICLE = /^(.+?),\s+(The|A|An)((?:\s+\(.*\))?)$/;

/**
 * The catalog uses MovieLens-style titles such as "Matrix Reloaded, The (2003)".
 * Show them the way people say them: "The Matrix Reloaded", with the year split out.
 */
export function parseTitle(raw: string | undefined): ParsedTitle {
  const m = raw?.match(/^(.*?)\s*\((\d{4})\)\s*$/);
  const base = (m ? m[1] : raw || "").trim();
  const article = base.match(TRAILING_ARTICLE);
  return {
    title: article ? `${article[2]} ${article[1]}${article[3]}` : base,
    year: m ? m[2] : null,
  };
}

/** "sci-fi" -> "Sci-Fi", "film-noir" -> "Film-Noir" */
export function prettyGenre(g: string): string {
  return g
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("-");
}
