"use client";

import { useState } from "react";

type CompressionResult = {
  filename: string;
  components: number;
  original_shape: number[];
  compressed_shape: number[];
  compression_ratio: number;
  reconstruction_error: number;
  relative_reconstruction_error: number;
  rmse: number;
  compressed_image: string;
};

export default function ImageCompression() {
  const [file, setFile] = useState<File | null>(null);
  const [components, setComponents] = useState(20);
  const [preview, setPreview] = useState("");
  const [result, setResult] =
    useState<CompressionResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  };

  const compressImage = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

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

      const formData = new FormData();

      formData.append("file", file);
      formData.append(
        "components",
        String(components),
      );

      const response = await fetch(
        `${apiUrl}/applications/image-compression`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Image compression failed.",
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
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setComponents(20);
  };

  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
          Application
        </p>

        <h2 className="mt-2 text-3xl font-bold">
          Image Compression
        </h2>

        <p className="mt-3 max-w-3xl leading-7 text-zinc-400">
          Compress images using low-rank SVD and
          explore the relationship between retained
          components, compression ratio, and
          reconstruction quality.
        </p>
      </div>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
        <h3 className="text-xl font-semibold">
          Configuration
        </h3>

        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Image
            </label>

            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleFileChange}
              className="block w-full rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm text-zinc-300 file:mr-4 file:rounded-md file:border-0 file:bg-white file:px-3 file:py-2 file:text-sm file:font-medium file:text-zinc-950"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              SVD Components
            </label>

            <input
              type="number"
              min={1}
              value={components}
              onChange={(event) =>
                setComponents(
                  Math.max(
                    1,
                    Number(event.target.value) || 1,
                  ),
                )
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-3 text-sm text-white outline-none focus:border-white"
            />

            <p className="mt-2 text-xs text-zinc-500">
              More components generally preserve more
              image detail while reducing compression.
            </p>
          </div>
        </div>

        {preview && (
          <div className="mt-6">
            <p className="mb-3 text-sm font-medium text-zinc-300">
              Original Image
            </p>

            <div className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-3">
              <img
                src={preview}
                alt="Selected image"
                className="max-h-[400px] w-auto rounded-lg object-contain"
              />
            </div>
          </div>
        )}

        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={compressImage}
            disabled={loading || !file}
            className="rounded-lg bg-white px-5 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? "Compressing..."
              : "Compress Image"}
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
        <>
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              label="Components"
              value={result.components.toString()}
            />

            <Metric
              label="Compression Ratio"
              value={`${result.compression_ratio.toFixed(2)}×`}
            />

            <Metric
              label="Relative Error"
              value={result.relative_reconstruction_error.toFixed(
                6,
              )}
            />

            <Metric
              label="RMSE"
              value={result.rmse.toFixed(4)}
            />
          </section>

          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
            <h3 className="text-xl font-semibold">
              Reconstruction
            </h3>

            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <ImageCard
                title="Original"
                src={preview}
                alt="Original image"
              />

              <ImageCard
                title="Compressed / Reconstructed"
                src={result.compressed_image}
                alt="Compressed reconstructed image"
              />
            </div>
          </section>

          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
            <h3 className="text-xl font-semibold">
              Compression Details
            </h3>

            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <Detail
                label="Original Shape"
                value={result.original_shape.join(" × ")}
              />

              <Detail
                label="Compressed Shape"
                value={result.compressed_shape.join(" × ")}
              />

              <Detail
                label="Reconstruction Error"
                value={result.reconstruction_error.toFixed(
                  4,
                )}
              />

              <Detail
                label="Relative Reconstruction Error"
                value={result.relative_reconstruction_error.toFixed(
                  6,
                )}
              />
            </div>
          </section>
        </>
      )}
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
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

      <p className="mt-2 font-medium text-zinc-300">
        {value}
      </p>
    </div>
  );
}

function ImageCard({
  title,
  src,
  alt,
}: {
  title: string;
  src: string;
  alt: string;
}) {
  return (
    <div>
      <p className="mb-3 text-sm font-medium text-zinc-300">
        {title}
      </p>

      <div className="flex min-h-[300px] items-center justify-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        {src ? (
          <img
            src={src}
            alt={alt}
            className="max-h-[500px] max-w-full rounded-lg object-contain"
          />
        ) : (
          <p className="text-sm text-zinc-600">
            Image unavailable
          </p>
        )}
      </div>
    </div>
  );
}
