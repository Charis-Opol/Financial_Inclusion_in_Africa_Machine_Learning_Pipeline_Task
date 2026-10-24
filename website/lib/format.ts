import type { MeanStd } from "@/data/content";

/** Three decimals, as every metric in the report is quoted. */
export const fixed3 = (n: number) => n.toFixed(3);

/** "0.590 ± 0.019" */
export const meanStd = (v: MeanStd) => `${fixed3(v.mean)} ± ${fixed3(v.std)}`;
