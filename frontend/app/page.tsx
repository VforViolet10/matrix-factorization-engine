
"use client";

import { useState } from "react";
import FactorizationPanel from "./components/FactorizationPanel";
import RecommendationSystem from "./components/RecommendationSystem";
import ImageCompression from "./components/ImageCompression";
import RecommendationEvaluation from "./components/RecommendationEvaluation";
import ThemeToggle from "./components/ThemeToggle";

type Application =
  | "factorization"
  | "recommendations"
  | "compression"
  | "evaluation";

const APPLICATIONS: {
  id: Application;
  number: string;
  label: string;
  description: string;
  symbol: string;
}[] = [
  {
    id: "factorization",
    number: "01",
    label: "Factorization",
    description: "Explore matrix decomposition methods.",
    symbol: "◈",
  },
  {
    id: "recommendations",
    number: "02",
    label: "Recommendations",
    description: "Discover latent patterns and predictions.",
    symbol: "✳",
  },
  {
    id: "compression",
    number: "03",
    label: "Image Compression",
    description: "Compress images using low-rank SVD.",
    symbol: "▧",
  },
  {
    id: "evaluation",
    number: "04",
    label: "Evaluation",
    description: "Measure recommendation performance.",
    symbol: "⌁",
  },
];

export default function Home() {
  const [activeApplication, setActiveApplication] =
    useState<Application>("factorization");

  return (
    <main className="app-shell">
      <div className="app-container">
        <header className="topbar">
          <a className="brand" href="#" aria-label="Matrix Factorization Engine home">
            <span className="brand-mark">M<span>×</span></span>
            <span className="brand-name">
              MATRIX<span>LAB</span>
            </span>
          </a>

          <div className="topbar-right">
            <span className="status-pill">
              <span className="status-dot" />
              Research workspace
            </span>
            <ThemeToggle />
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              LINEAR ALGEBRA · DATA SCIENCE · AI
            </div>

            <h1>
              Make sense of
              <br />
              <span className="gradient-text">the underlying</span>
              <br />
              structure.
            </h1>

            <p className="hero-description">
              An interactive research workspace for matrix
              decomposition, recommendation systems, image
              compression, and quantitative evaluation.
            </p>

            <div className="hero-meta">
              <span>Numerical methods</span>
              <span className="meta-separator">✳</span>
              <span>Low-rank modeling</span>
              <span className="meta-separator">✳</span>
              <span>Exploratory analysis</span>
            </div>
          </div>

          <div className="math-art" aria-hidden="true">
            <div className="math-orbit orbit-one" />
            <div className="math-orbit orbit-two" />
            <div className="math-grid">
              {Array.from({ length: 25 }, (_, i) => (
                <span
                  key={i}
                  className={`math-cell cell-${(i * 7 + 3) % 5}`}
                />
              ))}
            </div>
            <div className="math-symbol symbol-one">U</div>
            <div className="math-symbol symbol-two">Σ</div>
            <div className="math-symbol symbol-three">Vᵀ</div>
            <div className="math-caption">A = UΣVᵀ</div>
          </div>
        </section>

        <section className="workspace-section">
          <div className="section-heading">
            <div>
              <p className="section-kicker">YOUR TOOLKIT</p>
              <h2>Research workspace</h2>
              <p className="section-description">
                Choose a module to begin your exploration.
              </p>
            </div>
            <span className="module-count">04 MODULES</span>
          </div>

          <nav className="application-grid" aria-label="Applications">
            {APPLICATIONS.map((application) => {
              const isActive =
                activeApplication === application.id;

              return (
                <button
                  key={application.id}
                  type="button"
                  aria-pressed={isActive}
                  onClick={() =>
                    setActiveApplication(application.id)
                  }
                  className={`application-card accent-${application.id} ${
                    isActive ? "is-active" : ""
                  }`}
                >
                  <div className="application-card-top">
                    <span className="application-symbol">
                      {application.symbol}
                    </span>
                    <span className="application-number">
                      {application.number}
                    </span>
                  </div>

                  <span className="application-title">
                    {application.label}
                  </span>

                  <span className="application-description">
                    {application.description}
                  </span>

                  <span className="application-action">
                    {isActive ? "Currently selected" : "Open module"}
                    <span aria-hidden="true"> ↗</span>
                  </span>
                </button>
              );
            })}
          </nav>

          <div className="active-module-heading">
            <div className="active-module-indicator" />
            <span>ACTIVE MODULE</span>
            <span className="active-module-divider" />
            <strong>
              {APPLICATIONS.find(
                (item) => item.id === activeApplication
              )?.label}
            </strong>
          </div>

          <section className="module-panel">
            {activeApplication === "factorization" && (
              <FactorizationPanel />
            )}
            {activeApplication === "recommendations" && (
              <RecommendationSystem />
            )}
            {activeApplication === "compression" && (
              <ImageCompression />
            )}
            {activeApplication === "evaluation" && (
              <RecommendationEvaluation />
            )}
          </section>
        </section>

        <footer className="app-footer">
          <span>Matrix Factorization Engine</span>
          <span>Built for mathematical exploration.</span>
        </footer>
      </div>
    </main>
  );
}
