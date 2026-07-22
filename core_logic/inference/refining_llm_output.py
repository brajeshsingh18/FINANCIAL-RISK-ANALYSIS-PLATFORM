import re

def report_to_json(report):
    sections = {}
    current_heading = None
    current_content = []

    for line in report.splitlines():
        line = line.strip()
        if line.startswith("# "):
            if current_heading:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = (
                line[2:]
                .lower()
                .replace("&", "and")
                .replace(" ", "_")
            )
            current_content = []
        else:
            current_content.append(line)
    if current_heading:
        sections[current_heading] = "\n".join(current_content).strip()

    return sections