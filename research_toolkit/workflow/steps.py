
GRAMMARLY_GUIDE_STEPS = [
    {
        "step": 1,
        "name": "Understand the Assignment",
        "agent": "writer",
        "instruction": "Analyze the user's request. Identify the topic, length requirements (if any), and specific focus. Create a 'project_plan.txt' in the output folder summarizing the assignment."
    },
    {
        "step": 2,
        "name": "Choose Topic",
        "agent": "researcher",
        "instruction": "Perform a quick search to validate the topic. Ensure there is enough information available. If the topic is too broad, narrow it down. Report back with the refined topic."
    },
    {
        "step": 3,
        "name": "Gather Preliminary Research",
        "agent": "researcher",
        "instruction": "Search for key sources. Look for academic papers, credible articles, and data. Download or save the text of at least 3-5 key sources. Use `chrome_navigate` and `chrome_get_web_content`."
    },
    {
        "step": 4,
        "name": "Write Thesis Statement",
        "agent": "writer",
        "instruction": "Based on the preliminary research, write a strong thesis statement. It should be a single sentence that summarizes the main argument. Save it to 'thesis.txt'."
    },
    {
        "step": 5,
        "name": "Determine Supporting Evidence",
        "agent": "researcher",
        "instruction": "Go back to the sources. Find specific quotes, statistics, and facts that support the thesis. Extract this evidence and organize it by subtopic."
    },
    {
        "step": 6,
        "name": "Write Research Paper Outline",
        "agent": "writer",
        "instruction": "Create a detailed outline. Include Introduction (Thesis), Body Paragraphs (Subtopics + Evidence), and Conclusion. Save it to 'outline.txt'."
    },
    {
        "step": 7,
        "name": "Write First Draft",
        "agent": "writer",
        "instruction": "Write the full first draft of the paper based on the outline. Focus on getting the ideas down. Save it to 'draft.txt'."
    },
    {
        "step": 8,
        "name": "Cite Sources",
        "agent": "writer",
        "instruction": "Ensure all evidence in the draft is properly cited (APA format). Create a 'references.txt' file with the full bibliography."
    },
    {
        "step": 9,
        "name": "Edit and Proofread",
        "agent": "writer",
        "instruction": "Review the 'draft.txt'. Check for flow, clarity, and grammar. Create a final version named 'final_paper.docx' using the Word MCP tools (`create_document`, `add_heading`, `add_paragraph`). Ensure the document is well-structured."
    }
]
