"use client";

import { useState } from "react";

import MatrixEditor from "./MatrixEditor";

const DEFAULT_RATINGS = [
  [5, 4, 0, 0, 2, 1, 0, 0],
  [4, 5, 0, 0, 1, 2, 0, 0],
  [1, 2, 5, 4, 0, 0, 0, 1],
  [0, 1, 4, 5, 0, 0, 2, 1],
  [5, 4, 0, 0, 5, 4, 0, 0],
  [4, 5, 0, 0, 4, 5, 0, 0],
  [0, 0, 5, 4, 1, 0, 5, 4],
  [0, 0, 4, 5, 2, 1, 4, 5],
];

const MOVIES = [
  "Inception",
  "Interstellar",
  "The Dark Knight",
  "Dune",
  "Avengers",
  "Iron Man",
  "The Matrix",
  "Gladiator",
];

type Recommendation = {
  item: string;
  predicted_rating: number;
};

type RecommendationResponse = {
  user_index: number;
  method: string;
  recommendations: Recommendation[];
};

export default function RecommendationSystem() {
  const [ratings, setRatings] =
    useState<number[][]>(DEFAULT_RATINGS);

  const [method, setMethod] = useState("svd");
  const [components, setComponents] = useState(4);
  const [userIndex, setUserIndex] = useState(0);
  const [numRecommendations, setNumRecommendations] =
    useState(3);

  const [result, setResult] =
    useState<RecommendationResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generateRecommendations = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const apiUrl =
        process.env.NEXT_PUBLIC_API_URL;

      if (!apiUrl) {
        throw new Error(
          "NEXT_PUBLIC_API_URL is not configured.",
        );
      }

      const response = await fetch(
        `${apiUrl}/applications/recommendations`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ratings,
            method,
            n_components: components,
            user_index: userIndex,
            num_recommendations:
              numRecommendations,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Failed to generate recommendations.",
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong.",
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setRatings(
      DEFAULT_RATINGS.map((row) => [...row]),
    );
    setMethod("svd");
    setComponents(4);
    setUserIndex(0);
    setNumRecommendations(3);
    setResult(null);
    setError("");
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
          Application
        </p>

        <h2 className="mt-2 text-3xl font-bold">
          Recommendation System
        </h2>

        <p className="mt-3 max-w-3xl leading-7 text-zinc-400">
          Generate personalized recommendations
          from a user-item rating matrix using
          matrix factorization.
        </p>
      </div>

      {/* Rating Matrix */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <div className="mb-5">
          <h3 className="text-xl font-semibold">
            User-Item Rating Matrix
          </h3>

          <p className="mt-2 text-sm leading-6 text-zinc-500">
            Rows represent users and columns
            represent movies. A value of 0 represents
            an unrated item.
          </p>
        </div>

        <MatrixEditor
          matrix={ratings}
          onChange={setRatings}
        />
      </section>

      {/* Configuration */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <div className="mb-6">
          <h3 className="text-xl font-semibold">
            Recommendation Configuration
          </h3>

          <p className="mt-2 text-sm text-zinc-500">
            Configure the factorization model and
            recommendation parameters.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {/* Method */}
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Method
            </label>

            <select
              value={method}
              onChange={(event) =>
                setMethod(event.target.value)
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            >
              <option value="svd">
                SVD
              </option>
              <option value="nmf">
                NMF
              </option>
            </select>
          </div>

          {/* Components */}
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Components
            </label>

            <input
              type="number"
              min={1}
              max={8}
              value={components}
              onChange={(event) =>
                setComponents(
                  Math.max(
                    1,
                    Math.min(
                      8,
                      Number(event.target.value) || 1,
                    ),
                  ),
                )
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            />
          </div>

          {/* User */}
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              User
            </label>

            <select
              value={userIndex}
              onChange={(event) =>
                setUserIndex(
                  Number(event.target.value),
                )
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            >
              {ratings.map((_, index) => (
                <option
                  key={index}
                  value={index}
                >
                  User {index + 1}
                </option>
              ))}
            </select>
          </div>

          {/* Number of recommendations */}
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Recommendations
            </label>

            <input
              type="number"
              min={1}
              max={MOVIES.length}
              value={numRecommendations}
              onChange={(event) =>
                setNumRecommendations(
                  Math.max(
                    1,
                    Math.min(
                      MOVIES.length,
                      Number(event.target.value) || 1,
                    ),
                  ),
                )
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            />
          </div>
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={generateRecommendations}
            disabled={loading}
            className="rounded-lg bg-white px-5 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? "Generating..."
              : "Generate Recommendations"}
          </button>

          <button
            type="button"
            onClick={reset}
            className="rounded-lg border border-zinc-700 px-5 py-3 text-sm font-medium text-zinc-300 transition hover:bg-zinc-800"
          >
            Reset
          </button>
        </div>

        {error && (
          <div className="mt-5 rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}
      </section>

      {/* Movie Reference */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <h3 className="text-xl font-semibold">
          Dataset Items
        </h3>

        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {MOVIES.map((movie, index) => (
            <div
              key={movie}
              className="rounded-lg border border-zinc-800 bg-zinc-950 px-4 py-3"
            >
              <p className="text-xs text-zinc-600">
                Item {index}
              </p>

              <p className="mt-1 text-sm font-medium text-zinc-300">
                {movie}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Results */}
      {result && (
        <section className="rounded-2xl border border-zinc-700 bg-zinc-900 p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
                Results
              </p>

              <h3 className="mt-2 text-2xl font-bold">
                Recommendations for User{" "}
                {result.user_index + 1}
              </h3>

              <p className="mt-2 text-sm text-zinc-500">
                Method:{" "}
                <span className="font-medium uppercase text-zinc-300">
                  {result.method}
                </span>
              </p>
            </div>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {result.recommendations.map(
              (recommendation, index) => (
                <div
                  key={recommendation.item}
                  className="rounded-xl border border-zinc-800 bg-zinc-950 p-5"
                >
                  <p className="text-xs font-semibold uppercase tracking-wider text-zinc-600">
                    Recommendation {index + 1}
                  </p>

                  <h4 className="mt-3 text-lg font-semibold text-white">
                    {recommendation.item}
                  </h4>

                  <div className="mt-5">
                    <p className="text-xs text-zinc-500">
                      Predicted Rating
                    </p>

                    <p className="mt-1 text-3xl font-bold">
                      {recommendation.predicted_rating.toFixed(
                        2,
                      )}
                    </p>
                  </div>
                </div>
              ),
            )}
          </div>
        </section>
      )}
    </div>
  );
}
