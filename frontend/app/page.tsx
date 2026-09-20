"use client";

import { useState } from "react";

import FactorizationPanel from "./components/FactorizationPanel";
import RecommendationSystem from "./components/RecommendationSystem";
import ImageCompression from "./components/ImageCompression";
import RecommendationEvaluation from "./components/RecommendationEvaluation";

type Application =
  | "factorization"
  | "recommendations"
  | "compression"
  | "evaluation";

const APPLICATIONS: {
  id: Application;
  label: string;
  description: string;
}[] = [
  {
    id: "factorization",
    label: "Factorization",
    description:
      "Explore matrix decomposition methods.",
  },
  {
    id: "recommendations",
    label: "Recommendations",
    description:
      "Generate recommendations using matrix factorization.",
  },
  {
    id: "compression",
    label: "Image Compression",
    description:
      "Compress images using low-rank SVD.",
  },
  {
    id: "evaluation",
    label: "Evaluation",
    description:
      "Evaluate recommendation predictions.",
  },
];

export default function Home() {
  const [activeApplication, setActiveApplication] =
    useState<Application>("factorization");

  return (
    <main className="min-h-screen bg-zinc-950 text-white">
      <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8 lg:py-14">
        {/* Header */}
        <header className="mb-10">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-zinc-500">
            Linear Algebra · Data Science · Research
          </p>

          <h1 className="mt-3 text-4xl font-bold tracking-tight sm:text-5xl">
            Matrix Factorization Engine
          </h1>

          <p className="mt-5 max-w-3xl text-lg leading-8 text-zinc-400">
            An interactive platform for matrix
            decomposition, recommendation systems,
            image compression, and quantitative
            evaluation.
          </p>
        </header>

        {/* Application Navigation */}
        <nav
          aria-label="Applications"
          className="mb-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"
        >
          {APPLICATIONS.map((application) => {
            const isActive =
              activeApplication ===
              application.id;

            return (
              <button
                key={application.id}
                type="button"
                onClick={() =>
                  setActiveApplication(
                    application.id,
                  )
                }
                className={`rounded-2xl border p-5 text-left transition ${
                  isActive
                    ? "border-white bg-white text-zinc-950"
                    : "border-zinc-800 bg-zinc-900 text-white hover:border-zinc-600"
                }`}
              >
                <p className="font-semibold">
                  {application.label}
                </p>

                <p
                  className={`mt-2 text-sm leading-6 ${
                    isActive
                      ? "text-zinc-600"
                      : "text-zinc-500"
                  }`}
                >
                  {application.description}
                </p>
              </button>
            );
          })}
        </nav>

        {/* Active Application */}
        <section>
          {activeApplication ===
            "factorization" && (
            <FactorizationPanel />
          )}

          {activeApplication ===
            "recommendations" && (
            <RecommendationSystem />
          )}

          {activeApplication ===
            "compression" && (
            <ImageCompression />
          )}

          {activeApplication ===
            "evaluation" && (
            <RecommendationEvaluation />
          )}
        </section>
      </div>
    </main>
  );
}

function ComingSoon({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-8">
      <p className="text-sm font-semibold uppercase tracking-[0.15em] text-zinc-500">
        Application
      </p>

      <h2 className="mt-3 text-2xl font-bold">
        {title}
      </h2>

      <p className="mt-3 max-w-2xl leading-7 text-zinc-400">
        {description}
      </p>

      <div className="mt-6 inline-flex rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-400">
        Integration coming next
      </div>
    </div>
  );
}
