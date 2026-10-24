import Image from "next/image";
import type { ReportFigure as ReportFigureData } from "@/data/content";

/**
 * A figure exported from the project's notebooks. The plots have a light
 * background baked in, so they sit on a fixed "paper" backing in both
 * themes. Width/height are set to avoid layout shift; images lazy-load.
 */
export function ReportFigure({ figure, sizes }: { figure: ReportFigureData; sizes: string }) {
  return (
    <figure className="flex h-full flex-col rounded-lg border border-rule bg-surface p-3 sm:p-4">
      <div className="overflow-hidden rounded-md bg-[#fcfcfa]">
        <Image
          src={figure.src}
          width={figure.width}
          height={figure.height}
          alt={figure.alt}
          sizes={sizes}
          loading="lazy"
          className="block h-auto w-full"
        />
      </div>
      <figcaption className="mt-3 flex flex-1 flex-col justify-between gap-2 px-1 text-sm text-muted">
        <span>{figure.caption}</span>
        <a href={figure.src} className="self-start font-medium text-accent hover:underline">
          Open full size<span className="sr-only"> image</span>
          <span aria-hidden="true"> ↗</span>
        </a>
      </figcaption>
    </figure>
  );
}
