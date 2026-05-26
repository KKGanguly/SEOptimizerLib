const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial", color: "8B0000" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial", color: "333333" })]
  });
}

function para(text, options = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 22, font: "Arial", ...options })]
  });
}

function bullet(text, bold = false) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: "Arial", bold })]
  });
}

function timingRow(time, topic, notes, color = "FFFFFF") {
  return new TableRow({
    children: [
      new TableCell({
        borders,
        width: { size: 1200, type: WidthType.DXA },
        shading: { fill: color, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: time, size: 20, font: "Arial", bold: true })] })]
      }),
      new TableCell({
        borders,
        width: { size: 3000, type: WidthType.DXA },
        shading: { fill: color, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: topic, size: 20, font: "Arial", bold: true })] })]
      }),
      new TableCell({
        borders,
        width: { size: 5160, type: WidthType.DXA },
        shading: { fill: color, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: notes, size: 20, font: "Arial" })] })]
      }),
    ]
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    children: [

      // Title block
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "CSC316 — Data Structures & Algorithms", bold: true, size: 40, font: "Arial", color: "8B0000" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Lecture: Problems → Algorithms → Programs", size: 26, font: "Arial", color: "444444" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "Instructor Delivery Notes  |  Target: 35 minutes", size: 22, font: "Arial", color: "666666", italics: true })]
      }),
      new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "8B0000", space: 1 } },
        spacing: { after: 240 },
        children: []
      }),

      // Overview box
      new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Lecture Overview", bold: true, size: 26, font: "Arial", color: "8B0000" })] }),
      para("Three connected topics, each building on the last. Keep the Problem → Algorithm → Program arc visible throughout — students should leave understanding that these are distinct layers, not interchangeable terms."),
      new Paragraph({ spacing: { after: 240 }, children: [] }),

      // Timing table
      new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Timing at a Glance", bold: true, size: 26, font: "Arial", color: "8B0000" })] }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1200, 3000, 5160],
        rows: [
          new TableRow({
            children: [
              new TableCell({ borders, width: { size: 1200, type: WidthType.DXA }, shading: { fill: "8B0000", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Time", size: 20, font: "Arial", bold: true, color: "FFFFFF" })] })] }),
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "8B0000", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Topic", size: 20, font: "Arial", bold: true, color: "FFFFFF" })] })] }),
              new TableCell({ borders, width: { size: 5160, type: WidthType.DXA }, shading: { fill: "8B0000", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Notes", size: 20, font: "Arial", bold: true, color: "FFFFFF" })] })] }),
            ]
          }),
          timingRow("0–5 min",  "Intro & Definitions (Problem)", "Slides 2–3. Quick definitions only — problem, specification, instance.", "FFF8F8"),
          timingRow("5–13 min", "TSP Example (Problem)", "Slides 4–6. Walk the TSP carefully — students new to formal problem specs need time here.", "FFFFFF"),
          timingRow("13–20 min","Algorithms & Properties", "Slides 8–11. Cover greedy approach on TSP instance #1. Pause for questions.", "FFF8F8"),
          timingRow("20–24 min","Greedy Fails — TSP #2", "Slides 12–13. Key insight: correctness matters. Don't rush.", "FFFFFF"),
          timingRow("24–28 min","Pseudocode", "Slides 16–23. Live-demo the template. Show Java vs pseudocode side-by-side.", "FFF8F8"),
          timingRow("28–33 min","Data Structures & Programs", "Slides 25–32. Keep concise — these are grounding concepts.", "FFFFFF"),
          timingRow("33–35 min","Review & Wrap-up", "Slide 34. Hit the 4 key takeaways. Take 1–2 questions.", "FFF8F8"),
        ]
      }),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Section 1
      heading1("PART 1 — Problems (Slides 2–6)  [0–13 min]"),
      new Paragraph({ spacing: { after: 100 }, children: [] }),

      heading2("Slides 2–3: Definitions (~5 min)"),
      para("Open by drawing the Problem → Algorithm → Program diagram on the board before advancing slides. Ask students where they usually start when given an assignment (most say 'coding') — use this to motivate why starting with a precise problem definition matters."),
      new Paragraph({ spacing: { after: 100 }, children: [] }),
      para("Key distinctions to stress:", { bold: true }),
      bullet("Problem: general question with unspecified parameters"),
      bullet("Problem Specification: formal description of parameters + required solution properties"),
      bullet("Problem Instance: specific values plugged in — this is what an algorithm actually operates on"),
      new Paragraph({ spacing: { after: 180 }, children: [] }),

      heading2("Slides 4–6: Traveling Salesperson Problem (~8 min)"),
      para("The TSP is the primary running example for the whole lecture. Invest time here."),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("Talking points:", { bold: true }),
      bullet("Point out that the formal solution notation (the summation) looks scary but simply sums distances along a tour and adds the return leg."),
      bullet("On Slide 6 (Instance #1): walk through each distance. Ask: 'How many possible tours are there for 4 cities?' — answer is (4-1)! = 6, which you'll revisit with the exhaustive approach."),
      bullet("Make the parameters tangible: if C = {Raleigh, Durham, Chapel Hill, Cary}, what would d(c1, c2) represent?"),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("⚠ Common student confusion:", { bold: true, color: "AA0000" }),
      para("Students often conflate 'problem' and 'problem instance.' Return to this distinction whenever you demonstrate a new algorithm. A correct algorithm must solve ALL instances, not just the one on screen."),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Section 2
      heading1("PART 2 — Algorithms (Slides 8–15)  [13–24 min]"),
      new Paragraph({ spacing: { after: 100 }, children: [] }),

      heading2("Slides 8–10: Algorithm Definition & Properties (~4 min)"),
      para("Keep this brisk. The six properties (correctness, termination, robustness, complexity, adaptability, efficiency) are reference material — students do not need to memorize all six today."),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("Prioritize correctness and efficiency — those drive the rest of the lecture."),
      new Paragraph({ spacing: { after: 160 }, children: [] }),

      heading2("Slides 11–13: Greedy TSP — Instance #1 then Instance #2 (~7 min)"),
      para("Slide 11 (Greedy on Instance #1): Walk step-by-step. Total = 27. Ask: 'Is this optimal?' Students often say yes — set up the payoff."),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("Slides 12–13 (Greedy on Instance #2): This is the lecture's 'aha' moment.", { bold: true }),
      bullet("Greedy gives 108. Exhaustive finds routes costing 11 (Slide 14)."),
      bullet("Greedy is wrong — it violates correctness. Emphasize: an algorithm that fails on even one instance is not a correct algorithm for the problem."),
      bullet("Ask: 'What changed between instance #1 and instance #2?' The extreme distance (100) forced the greedy algorithm into a bad early choice it couldn't recover from."),
      new Paragraph({ spacing: { after: 160 }, children: [] }),

      heading2("Slide 15: Correctness & Efficiency (~3 min)"),
      para("Bridge from the TSP discussion to formal algorithm properties. The exhaustive approach is correct but scales as O(n!) — for 20 cities that's ~2.4 quintillion routes. Efficiency is not optional."),
      para("End with the callout: algorithms should be documented in pseudocode, not in a programming language. This is the bridge to Part 3."),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Section 3
      heading1("PART 3 — Pseudocode (Slides 16–23)  [24–28 min]"),
      new Paragraph({ spacing: { after: 100 }, children: [] }),

      heading2("Slides 16–20: What is Pseudocode & the CSC316 Format (~3 min)"),
      para("The key points: pseudocode is language-agnostic, human-readable, and the format used in this course for grading. Refer students to the reference sheet on Moodle."),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("Notation highlights to call out on Slide 20:", { bold: true }),
      bullet("← for assignment (not = which means comparison in many languages)"),
      bullet("for i ← X to Y do — both bounds inclusive"),
      bullet("Indentation, not braces, defines scope"),
      new Paragraph({ spacing: { after: 160 }, children: [] }),

      heading2("Slides 21–23: Live Demo (~2 min)"),
      para("Write the getMinimumGrade pseudocode live (Slide 22) while the Java version is visible. Then show Slide 23's bad example and ask students what's wrong before revealing the X marks."),
      para("The 'not clear or specific' example resonates well — students have all submitted vague pseudocode. Reinforce: a reader must be able to implement your pseudocode in any language without asking you questions."),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Section 4
      heading1("PART 4 — Data Structures & Programs (Slides 25–32)  [28–33 min]"),
      new Paragraph({ spacing: { after: 100 }, children: [] }),

      heading2("Slides 25–29: Data Structures (~3 min)"),
      para("Use the list examples on Slide 27 as a quick discussion — 'Who can tell me when a linked list beats an array?' These scenarios foreshadow the course topics students will spend weeks on."),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      para("Briefly define static vs dynamic and deterministic vs probabilistic. Students have seen arrays and may know ArrayList; the linked list terminology may be new."),
      new Paragraph({ spacing: { after: 160 }, children: [] }),

      heading2("Slides 31–33: Programs & the SDLC (~2 min)"),
      para("A program is just an implementation of an algorithm in a specific language. The SDLC diagram (Slide 33) maps directly: Requirements = Problem, Design = Algorithm, Implementation = Program. This is the engineering framing for everything they'll do this semester."),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Wrap-up
      heading1("WRAP-UP (Slide 34)  [33–35 min]"),
      new Paragraph({ spacing: { after: 100 }, children: [] }),
      para("Read the four takeaways aloud and briefly expand on each:"),
      bullet("Algorithms must solve ALL instances — revisit greedy TSP failure"),
      bullet("Pseudocode is the required format — not Java, not English prose"),
      bullet("Any programmer should translate your pseudocode into any language"),
      bullet("Data structure choice affects algorithm and program efficiency"),
      new Paragraph({ spacing: { after: 160 }, children: [] }),
      para("Leave 1–2 minutes for questions. If time is tight, skip SDLC slide and jump straight to Slide 34."),
      new Paragraph({ spacing: { after: 280 }, children: [] }),

      // Tips box
      new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "8B0000", space: 1 } },
        spacing: { after: 160 },
        children: [new TextRun({ text: "Pacing Tips & Contingencies", bold: true, size: 26, font: "Arial", color: "8B0000" })]
      }),
      bullet("Running short: Spend more time on Slides 12–14 (greedy failure). This is the richest conceptual moment."),
      bullet("Running long: Cut SDLC slide (33) and the deterministic/probabilistic distinction (Slide 29) — both are review-able independently."),
      bullet("Student engagement: After Slide 13, ask 'Can you think of a real-world problem where greedy is good enough?' before moving on."),
      bullet("If pseudocode notation feels rushed, remind students the reference sheet covers it — they don't need to memorize it from slides."),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/mnt/user-data/outputs/CSC316_Instructor_Notes.docx", buffer);
  console.log("Done");
});