"use client";

import { useState } from "react";

const TRAIN_RATINGS = [
  [5, 4, 0, 0, 2, 1, 0, 0],
  [4, 5, 0, 0, 1, 2, 0, 0],
  [1, 2, 5, 4, 0, 0, 0, 1],
  [0, 1, 4, 5, 0, 0, 2, 1],
  [5, 4, 0, 0, 5, 4, 0, 0],
  [4, 5, 0, 0, 4, 5, 0, 0],
  [0, 0, 5, 4, 1, 0, 5, 4],
  [0, 0, 4, 5, 2, 1, 4, 5],
];

const TEST_RATINGS = [
  [0, 0, 0, 0, 0, 0, 5, 4],
  [0, 0, 0, 0, 0, 0, 4, 5],
  [0, 0, 0, 0, 5, 4, 0, 0],
  [0, 0, 0, 0, 5, 4, 0, 0],
  [0, 0, 5, 4, 0, 0, 0, 0],
  [0, 0, 4, 5, 0, 0, 0, 0],
  [5, 4, 0, 0, 0, 0, 0, 0],
  [4, 5, 0, 0, 0, 0, 0, 0],
];

const PREDICTIONS = [
  [0, 0, 0, 0, 0, 0, 4.6, 4.1],
  [0, 0, 0, 0, 0, 0, 4.0, 4.7],
  [0, 0, 0, 0, 4.5, 4.2, 0, 0],
  [0, 0, 0, 0, 4.1, 4.6, 0, 0],
  [0, 0, 4.4, 4.0, 0, 0, 0, 0],
  [0, 0, 4.1, 4.5, 0, 0, 0, 0],
  [4.7, 4.2, 0, 0, 0, 0, 0, 0],
  [4.1, 4.6, 0, 0, 0, 0, 0, 0],
];

type EvaluationResult = {
  rmse: number;
  precision_at_k: number;
  recall_at_k: number;
  ndcg_at_k: number;
  k: number;
  relevance_threshold: number;
};

export default function RecommendationEvaluation() {
  const [predictions, setPredictions] =
    useState<number[][]>(
      PREDICTIONS.map((row) => [...row]),
    );

  const [trainRatings, setTrainRatings] =
    useState<number[][]>(
      TRAIN_RATINGS.map((row) => [...row]),
    );

  const [testRatings, setTestRatings] =
    useState<number[][]>(
      TEST_RATINGS.map((row) => [...row]),
    );

  const [k, setK] = useState(2);
  const [relevanceThreshold, setRelevanceThreshold] =
    useState(4);

  const [result, setResult] =
    useState<EvaluationResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const evaluate = async () => {
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
        `${apiUrl}/applications/recommendations/evaluate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            predictions,
            train_ratings: trainRatings,
            test_ratings: testRatings,
            k,
            relevance_threshold:
              relevanceThreshold,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Evaluation failed.",
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
    setPredictions(
      PREDICTIONS.map((row) => [...row]),
    );

    setTrainRatings(
      TRAIN_RATINGS.map((row) => [...row]),
    );

    setTestRatings(
      TEST_RATINGS.map((row) => [...row]),
    );

    setK(2);
    setRelevanceThreshold(4);
    setResult(null);
    setError("");
  };

  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
          Application
        </p>

        <h2 className="mt-2 text-3xl font-bold">
          Recommendation Evaluation
        </h2>

        <p className="mt-3 max-w-3xl leading-7 text-zinc-400">
          Evaluate recommendation predictions against
          held-out test ratings using RMSE,
          Precision@K, Recall@K, and NDCG@K.
        </p>
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
        <p className="text-sm leading-6 text-zinc-400">
          The default dataset uses a simulated
          train/test split. Training ratings represent
          information available to the model, while
          test ratings represent held-out ground truth.
        </p>
      </div>

      <MatrixInput
        title="Predictions"
        description="Predicted ratings for the held-out items."
        matrix={predictions}
        onChange={setPredictions}
      />

      <MatrixInput
        title="Training Ratings"
        description="Ratings available to the recommendation model."
        matrix={trainRatings}
        onChange={setTrainRatings}
      />

      <MatrixInput
        title="Test Ratings"
        description="Held-out ratings used as ground truth."
        matrix={testRatings}
        onChange={setTestRatings}
      />

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <h3 className="text-xl font-semibold">
          Evaluation Configuration
        </h3>

        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              K
            </label>

            <input
              type="number"
              min={1}
              value={k}
              onChange={(event) =>
                setK(
                  Math.max(
                    1,
                    Number(event.target.value) || 1,
                  ),
                )
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Relevance Threshold
            </label>

            <input
              type="number"
              min={0}
              max={5}
              step={0.5}
              value={relevanceThreshold}
              onChange={(event) =>
                setRelevanceThreshold(
                  Math.max(
                    0,
                    Number(event.target.value) || 0,
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
            onClick={evaluate}
            disabled={loading}
            className="rounded-lg bg-white px-5 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? "Evaluating..."
              : "Evaluate Predictions"}
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

      {result && (
        <section className="rounded-2xl border border-zinc-700 bg-zinc-900 p-6">
          <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
            Results
          </p>

          <h3 className="mt-2 text-2xl font-bold">
            Evaluation Metrics
          </h3>

          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              label="RMSE"
              value={result.rmse.toFixed(4)}
            />

            <Metric
              label="Precision@K"
              value={result.precision_at_k.toFixed(4)}
            />

            <Metric
              label="Recall@K"
              value={result.recall_at_k.toFixed(4)}
            />

            <Metric
              label="NDCG@K"
              value={result.ndcg_at_k.toFixed(4)}
            />
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <Detail
              label="K"
              value={result.k.toString()}
            />

            <Detail
              label="Relevance Threshold"
              value={result.relevance_threshold.toString()}
            />
          </div>
        </section>
      )}
    </div>
  );
}

function MatrixInput({
  title,
  description,
  matrix,
  onChange,
}: {
  title: string;
  description: string;
  matrix: number[][];
  onChange: (matrix: number[][]) => void;
}) {
  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
      <h3 className="text-xl font-semibold">
        {title}
      </h3>

      <p className="mt-2 text-sm leading-6 text-zinc-500">
        {description}
      </p>

      <div className="mt-5">
        <SimpleMatrixEditor
          matrix={matrix}
          onChange={onChange}
        />
      </div>
    </section>
  );
}

function SimpleMatrixEditor({
  matrix,
  onChange,
}: {
  matrix: number[][];
  onChange: (matrix: number[][]) => void;
}) {
  const updateCell = (
    rowIndex: number,
    columnIndex: number,
    value: string,
  ) => {
    const next = matrix.map((row) => [...row]);
    const parsed = Number(value);

    next[rowIndex][columnIndex] =
      value === "" || Number.isNaN(parsed)
        ? 0
        : parsed;

    onChange(next);
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4">
      <table className="border-collapse">
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((value, columnIndex) => (
                <td
                  key={columnIndex}
                  className="p-1"
                >
                  <input
                    type="number"
                    value={value}
                    onChange={(event) =>
                      updateCell(
                        rowIndex,
                        columnIndex,
                        event.target.value,
                      )
                    }
                    className="h-11 w-20 rounded-lg border border-zinc-700 bg-zinc-900 px-2 text-center text-sm text-white outline-none focus:border-white"
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
      <p className="text-sm text-zinc-500">
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold">
        {value}
      </p>
    </div>
  );
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
      <p className="text-xs uppercase tracking-wider text-zinc-600">
        {label}
      </p>

      <p className="mt-2 font-semibold text-zinc-300">
        {value}
      </p>
    </div>
  );
}
