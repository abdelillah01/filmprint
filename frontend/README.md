# Frontend

Next.js 16 (App Router) + React 19 + TypeScript + Tailwind 3 + framer-motion, Geist / Geist Mono. A single page: search a film, get ten similar ones.

```bash
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm install
npm run dev                  # http://localhost:3000 (webpack)
```

The backend must be running; its URL comes from `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`) and the backend's CORS list only allows `localhost:3000` / `127.0.0.1:3000` by default.

| File | Role |
|---|---|
| `app/page.tsx` | page state: loading, results, errors, "no exact match" banner, featured-film chips |
| `components/` | `SearchBar`, `MovieCard`, `LoadingSkeleton` |
| `lib/title.ts` | turns catalog titles (`Matrix, The (1999)`) into display form (`The Matrix`, year split out) |
| `lib/api.ts` | typed client for `POST /api/v1/recommend/`; `ApiError` + `describeError` turn 404 / 429 / 503 / network failures into distinct messages |

The `query` field in the API response is present only when the backend fuzzily corrected the input; the banner is driven by it. Keep the featured chips to titles that exist in the catalog (it ends in 2016).

```bash
npm run lint
npx tsc --noEmit
npm run build
```

## Design tokens

Graphite neutrals and one ember accent live as CSS variables in `app/globals.css` and are mapped to Tailwind colors in `tailwind.config.js` (`bg`, `card`, `raised`, `line`, `ink`, `ink-2`, `ink-muted`, `accent`). Surfaces stay neutral so the posters carry the color; no gradients, glows, glass or looping motion. Design pass follows the [unslop](https://github.com/heydamir/unslop) guidelines by Damir Sokolovsky (CC BY 4.0).
