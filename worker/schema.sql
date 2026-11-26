-- Human ratings (D-043). Anonymous: a random rater id per browser, no IP, no names.
CREATE TABLE IF NOT EXISTS ratings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app TEXT NOT NULL,
  rater TEXT NOT NULL,
  usefulness INTEGER NOT NULL CHECK (usefulness BETWEEN 1 AND 5),
  novelty INTEGER NOT NULL CHECK (novelty BETWEEN 1 AND 5),
  works TEXT NOT NULL CHECK (works IN ('yes', 'partly', 'no')),
  seconds REAL,
  lang TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (app, rater)
);
CREATE INDEX IF NOT EXISTS ratings_app ON ratings (app);
