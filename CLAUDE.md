# Claude Development Guide

## Project Overview
This is a website project with blog posts and glossary content, including Jupyter notebooks for technical documentation.

## Jupyter Notebook Guidelines
- Notebooks are located in the `glossary/` directory
- Use descriptive cell types (markdown for explanations, code for examples)
- Include proper imports and version checking in code cells
- Add clear section headers with emojis for visual appeal
- Test code cells to ensure they run without errors

## Common Commands
- To work with notebooks: Use NotebookRead and NotebookEdit tools
- To search for files: Use Glob tool with patterns like `**/*.ipynb`
- To check project structure: Use LS tool

## Testing
- Always verify notebook cells execute properly
- Check that imports are available in the environment
- Ensure markdown cells render correctly

## Style Guidelines
- Use clear, descriptive section headers
- Add helpful comments in code cells
- Include version information for key libraries
- Use consistent formatting across notebooks