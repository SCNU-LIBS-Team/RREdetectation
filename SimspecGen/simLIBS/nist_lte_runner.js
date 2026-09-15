"use strict";

// Run NIST's browser-side Saha/LTE calculation without starting a browser.
// The Python caller supplies the official JavaScript and the data declarations
// embedded in the NIST result page through stdin.
const fs = require("node:fs");
const vm = require("node:vm");

function fail(message) {
  process.stderr.write(String(message) + "\n");
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(0, "utf8"));
} catch (error) {
  fail(`Unable to read static LIBS input: ${error.message}`);
}

const context = {
  // The calculation only checks these DOM fields to show warnings/titles.
  document: {
    documentMode: false,
    getElementById: () => null,
  },
  window: { StyleMedia: false },
  performance: { now: () => 0 },
  alert: (message) => {
    throw new Error(String(message));
  },
};
vm.createContext(context);

try {
  vm.runInContext(input.lteSource, context, {
    filename: "nist-saha_lte.js",
    timeout: input.timeoutMilliseconds,
  });
  vm.runInContext(input.dataSource, context, {
    filename: "nist-libs-data.js",
    timeout: input.timeoutMilliseconds,
  });

  // NIST estimates browser memory by allocating a very large string. It only
  // uses the result to lower the requested resolution. The headless process is
  // intentionally allowed to use the resolution requested by the caller.
  context.memAvail = () => Number.MAX_SAFE_INTEGER;
  context.resolution = Number(input.resolution);
  context.composition = input.composition;
  context.temp = Number(input.temperature);
  context.eden = Number(input.electronDensity);

  const result = context.lte_spectrum_js(
    context.spectra,
    context.composition,
    context.lines,
    context.levels,
    context.ips,
    null,
    Number(context.temp),
    Number(context.eden),
    context.resolution,
    Number(context.cmeV),
    Number(context.koeff),
    "Wavelength (nm)",
    context.dataDopplerArray,
  );

  const rows = result[0];
  if (!Array.isArray(rows) || rows.length < 2) {
    throw new Error("NIST calculation returned no Doppler spectrum");
  }

  const headers = rows[0].map((item) =>
    item && typeof item === "object" ? item.label : item,
  );
  const sumIndex = headers.indexOf("Sum(calc)");
  if (sumIndex < 0) {
    throw new Error(
      `NIST calculation did not return a Sum(calc) column: ${headers.join(", ")}`,
    );
  }

  // Mirror NIST's arrToRealCSV() for every returned ion column. Calculated
  // intensities use Number.toExponential(3); absent ion values remain blank.
  const csv = rows.map((row, rowIndex) =>
    row
      .map((rawItem, columnIndex) => {
        let item = rawItem;
        if (rowIndex === 0 && item && typeof item === "object") {
          item = item.label;
        } else if (item === null || item === undefined) {
          item = "";
        }
        if (
          rowIndex > 0 &&
          columnIndex > 0 &&
          item !== "" &&
          item != "0"
        ) {
          item = Number(item).toExponential(3);
        }
        item = String(item);
        return item.includes(",") ? `"${item}"` : item;
      })
      .join(","),
  );
  process.stdout.write(csv.join("\n"));
} catch (error) {
  fail(`Static NIST Saha/LTE calculation failed: ${error.stack || error.message}`);
}
