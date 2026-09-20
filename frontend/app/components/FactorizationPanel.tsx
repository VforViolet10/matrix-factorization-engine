"use client";

import { useState } from "react";

import MatrixEditor from "./MatrixEditor";

const DEFAULT_MATRIX = [
  [5, 3, 1, 4],
  [4, 2, 1, 5],
  [1, 5, 4, 2],
  [5, 4, 2, 3],
];

const METHODS = [
  "svd",
  "nmf",
  "qr",
  "lu",
  "eigen",
  "incremental_svd",
  "robust_pca",
];

const COMPONENT_METHODS = new Set([
  "svd",
  "nmf",
  "incremental_svd",
]);

type FactorizationResult = {
  method: string;
  matrix_shape: number[];
  n_components: number | null;
  input_matrix: number[][];
  reconstructed_matrix: number[][];
  reconstruction_error: number | null;
  runtime_ms: number;
};

function getMatrixStats(matrix: number[][]) {
  const values = matrix.flat();

  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;

  if (values.length === 0) {
    return {
      rows,
      columns,
      elements: 0,
      rank: 0,
      min: 0,
      max: 0,
      mean: 0,
      frobeniusNorm: 0,
    };
  }

  const sum = values.reduce(
    (total, value) => total + value,
    0,
  );

  const frobeniusNorm = Math.sqrt(
    values.reduce(
      (total, value) => total + value ** 2,
      0,
    ),
  );

  let rank = 0;

  try {
    const tolerance = 1e-10;
    const a = matrix.map((row) => [...row]);

    const m = rows;
    const n = columns;

    let pivotRow = 0;

    for (
      let column = 0;
      column < n && pivotRow < m;
      column++
    ) {
      let pivot = pivotRow;

      for (
        let i = pivotRow + 1;
        i < m;
        i++
      ) {
        if (
          Math.abs(a[i][column]) >
          Math.abs(a[pivot][column])
        ) {
          pivot = i;
        }
      }

      if (
        Math.abs(a[pivot][column]) <=
        tolerance
      ) {
        continue;
      }

      [a[pivotRow], a[pivot]] = [
        a[pivot],
        a[pivotRow],
      ];

      for (
        let i = pivotRow + 1;
        i < m;
        i++
      ) {
        const factor =
          a[i][column] /
          a[pivotRow][column];

        for (
          let j = column;
          j < n;
          j++
        ) {
          a[i][j] -=
            factor *
            a[pivotRow][j];
        }
      }

      rank++;
      pivotRow++;
    }
  } catch {
    rank = 0;
  }

  return {
    rows,
    columns,
    elements: values.length,
    rank,
    min: Math.min(...values),
    max: Math.max(...values),
    mean: sum / values.length,
    frobeniusNorm,
  };
}

export default function FactorizationPanel() {
  const [matrix, setMatrix] =
    useState(DEFAULT_MATRIX);

  const [method, setMethod] =
    useState("svd");

  const [nComponents, setNComponents] =
    useState(2);

  const [result, setResult] =
    useState<FactorizationResult | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const supportsComponents =
    COMPONENT_METHODS.has(method);

  const stats =
    getMatrixStats(matrix);

  const maxComponents = Math.min(
    matrix.length,
    matrix[0]?.length ?? 1,
  );

  const handleMethodChange = (
    nextMethod: string,
  ) => {
    setMethod(nextMethod);
    setResult(null);
    setError("");
  };

  const runFactorization = async () => {
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

      const payload: {
        matrix: number[][];
        method: string;
        n_components?: number;
      } = {
        matrix,
        method,
      };

      if (supportsComponents) {
        payload.n_components =
          Math.min(
            nComponents,
            maxComponents,
          );
      }

      const response = await fetch(
        `${apiUrl}/factorize`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Factorization failed.",
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the backend.",
      );
    } finally {
      setLoading(false);
    }
  };

  const resetMatrix = () => {
    setMatrix(
      DEFAULT_MATRIX.map((row) => [
        ...row,
      ]),
    );

    setResult(null);
    setError("");
    setMethod("svd");
    setNComponents(2);
  };

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <section>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-zinc-500">
          Linear Algebra
        </p>

        <h2 className="mt-2 text-3xl font-bold tracking-tight">
          Matrix Factorization
        </h2>

        <p className="mt-3 max-w-3xl text-zinc-400">
          Experiment with matrix decomposition
          methods, adjust the input matrix, and
          inspect reconstruction quality and
          runtime.
        </p>
      </section>

      {/* Matrix Statistics */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <div className="mb-5">
          <h3 className="text-xl font-semibold">
            Matrix Statistics
          </h3>

          <p className="mt-1 text-sm text-zinc-500">
            Live statistics calculated from the
            current input matrix.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
          <StatCard
            label="Dimensions"
            value={`${stats.rows} × ${stats.columns}`}
          />

          <StatCard
            label="Elements"
            value={String(stats.elements)}
          />

          <StatCard
            label="Rank"
            value={String(stats.rank)}
          />

          <StatCard
            label="Minimum"
            value={stats.min.toFixed(3)}
          />

          <StatCard
            label="Maximum"
            value={stats.max.toFixed(3)}
          />

          <StatCard
            label="Mean"
            value={stats.mean.toFixed(3)}
          />

          <StatCard
            label="Frobenius Norm"
            value={stats.frobeniusNorm.toFixed(3)}
          />
        </div>
      </section>

      {/* Input + Configuration */}
      <section className="grid gap-6 lg:grid-cols-[1.4fr_0.6fr]">
        {/* Matrix Editor */}
        <div className="rounded-2xl bg-white p-6 text-zinc-950 shadow-2xl">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h3 className="text-xl font-semibold">
                Input Matrix
              </h3>

              <p className="mt-1 text-sm text-zinc-500">
                Edit values or change the matrix
                dimensions.
              </p>
            </div>

            <button
              type="button"
              onClick={resetMatrix}
              className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
            >
              Reset
            </button>
          </div>

          <MatrixEditor
            matrix={matrix}
            onChange={setMatrix}
          />
        </div>

        {/* Configuration */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
          <h3 className="text-xl font-semibold">
            Configuration
          </h3>

          <div className="mt-6 space-y-5">
            <div>
              <label
                htmlFor="factorization-method"
                className="mb-2 block text-sm font-medium text-zinc-300"
              >
                Factorization Method
              </label>

              <select
                id="factorization-method"
                value={method}
                onChange={(event) =>
                  handleMethodChange(
                    event.target.value,
                  )
                }
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
              >
                {METHODS.map((item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item.toUpperCase()}
                  </option>
                ))}
              </select>

              <p className="mt-2 text-xs text-zinc-500">
                {supportsComponents
                  ? "This method supports reduced-rank components."
                  : "This method does not require a component count."}
              </p>
            </div>

            {supportsComponents && (
              <div>
                <label
                  htmlFor="factorization-components"
                  className="mb-2 block text-sm font-medium text-zinc-300"
                >
                  Components
                </label>

                <input
                  id="factorization-components"
                  type="number"
                  min={1}
                  max={maxComponents}
                  value={nComponents}
                  onChange={(event) =>
                    setNComponents(
                      Number(
                        event.target.value,
                      ),
                    )
                  }
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
                />

                <p className="mt-2 text-xs text-zinc-500">
                  Maximum:{" "}
                  {maxComponents}
                </p>
              </div>
            )}

            <button
              type="button"
              onClick={runFactorization}
              disabled={loading}
              className="w-full rounded-lg bg-white px-4 py-3 font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading
                ? "Running..."
                : "Run Factorization"}
            </button>

            {error && (
              <div className="rounded-lg border border-red-900 bg-red-950/50 p-4 text-sm text-red-300">
                {error}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Results */}
      {result && (
        <section className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Method"
              value={result.method.toUpperCase()}
            />

            <StatCard
              label="Matrix Shape"
              value={`${result.matrix_shape[0]} × ${result.matrix_shape[1]}`}
            />

            <StatCard
              label="Reconstruction Error"
              value={
                result.reconstruction_error !==
                null
                  ? result.reconstruction_error.toFixed(
                      6,
                    )
                  : "N/A"
              }
            />

            <StatCard
              label="Runtime"
              value={`${result.runtime_ms.toFixed(
                2,
              )} ms`}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <MatrixDisplay
              title="Original Matrix"
              matrix={result.input_matrix}
            />

            <MatrixDisplay
              title="Reconstructed Matrix"
              matrix={
                result.reconstructed_matrix
              }
            />
          </div>
        </section>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
      <p className="text-sm text-zinc-500">
        {label}
      </p>

      <p className="mt-2 text-xl font-semibold">
        {value}
      </p>
    </div>
  );
}

function MatrixDisplay({
  title,
  matrix,
}: {
  title: string;
  matrix: number[][];
}) {
  return (
    <div className="rounded-2xl bg-white p-6 text-zinc-950">
      <h3 className="mb-4 text-lg font-semibold">
        {title}
      </h3>

      <div className="overflow-x-auto rounded-xl border border-zinc-200 p-4">
        <table className="border-collapse">
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map(
                  (value, columnIndex) => (
                    <td
                      key={columnIndex}
                      className="min-w-20 border border-zinc-200 px-3 py-3 text-center text-sm"
                    >
                      {Number(
                        value,
                      ).toFixed(3)}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
