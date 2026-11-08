"""Rebuild the live preprint (paper/generated/* and paper/main.pdf) from the current experiment results."""

from affdrift.reporting.paper import build

if __name__ == "__main__":
    result = build()
    print("pdf:", result["pdf"])
    for f in result["figures_failed"]:
        print("figure failed:", f)
    if "latex_error" in result:
        print(result["latex_error"])
