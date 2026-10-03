"use client";

import { useEffect } from "react";
import Link from "next/link";

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Application error:", error);
  }, [error]);

  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center justify-center p-8">
      <div className="max-w-md w-full bg-red-500/10 border border-red-500/20 rounded-2xl p-8 text-center space-y-6">
        <div className="text-4xl">⚠️</div>
        <div>
          <h2 className="text-xl font-bold text-red-400 mb-2">Something went wrong!</h2>
          <p className="text-sm text-white/60 mb-6">
            We apologize, but an unexpected error occurred. Our team has been notified.
          </p>
          {process.env.NODE_ENV === "development" && (
            <div className="bg-black/50 p-4 rounded text-left text-xs text-red-300 font-mono overflow-auto mb-6 max-h-48">
              {error.message}
            </div>
          )}
        </div>
        <div className="flex gap-4 justify-center">
          <button
            onClick={() => reset()}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-sm transition"
          >
            Try again
          </button>
          <Link
            href="/dashboard"
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm transition"
          >
            Return to Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
