
import os
from smolagents import Tool

class WriteFileTool(Tool):
    name = "write_file"
    description = "Writes content to a file in the output directory."
    inputs = {
        "filename": {"type": "string", "description": "The name of the file to write (e.g., 'outline.txt')."},
        "content": {"type": "string", "description": "The text content to write to the file."}
    }
    output_type = "string"

    def forward(self, filename: str, content: str) -> str:
        # Security: Ensure we only write to the output directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
        file_path = os.path.join(base_dir, filename)
        
        # Basic path traversal check
        if not os.path.abspath(file_path).startswith(base_dir):
            return "Error: Cannot write outside the output directory."

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

class CreateDirectoryTool(Tool):
    name = "create_directory"
    description = "Creates a new directory in the output folder."
    inputs = {
        "dirname": {"type": "string", "description": "The name of the directory to create."}
    }
    output_type = "string"

    def forward(self, dirname: str) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
        dir_path = os.path.join(base_dir, dirname)
        
        if not os.path.abspath(dir_path).startswith(base_dir):
            return "Error: Cannot create directory outside the output directory."

        try:
            os.makedirs(dir_path, exist_ok=True)
            return f"Successfully created directory {dirname}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"
