import os
import tempfile

def save_uploaded_file(uploaded_file) -> str:
    """Save streamlit uploaded file to disk and return path."""
    upload_dir = "data/upload_docs"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    print(f"✅ File saved: {file_path}")
    return file_path

def format_sources(sources: list) -> str:
    """Format source list into readable string."""
    if not sources:
        return "No sources available."
    return "\n".join([f"• {source}" for source in sources])
