import os
import subprocess

import requests


def search_and_download_books(query):
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": query,
        "output": "json",
        "fl[]": "identifier,title,creator,date,mediatype",
    }

    response = requests.get(url, params=params)
    data = response.json()
    os.makedirs("books", exist_ok=True)
    # Download the first 5 books found
    for result in data['response']['docs'][:5]:
        identifier = result.get('identifier')
        metadata_url = f"https://archive.org/metadata/{identifier}"
        metadata_response = requests.get(metadata_url)
        metadata = metadata_response.json()
        file_path = os.path.join("books", f"{identifier}")
        files = metadata['files']
        # Find and download the first PDF file associated with the item
        pdf_files = [file['name'] for file in files if file['name'].endswith('.pdf')]
        if pdf_files:
            pdf_url = f"https://archive.org/download/{identifier}/{pdf_files[0]}"
            pdf_response = requests.get(pdf_url)
            with open(f"{file_path}.pdf", "wb") as pdf_file:
                pdf_file.write(pdf_response.content)
            print(f"Downloaded: {file_path}.pdf")
            open_pdf(f"{file_path}.pdf")

def open_pdf(file_name):
    """
    Open the specified PDF file using the default PDF viewer on the system.
    """
    try:
        if os.name == 'nt':  # For Windows
            os.startfile(file_name)
        elif os.name == 'posix':  # For Unix/Linux/MacOS
            subprocess.run(['xdg-open', file_name])
        else:
            print("Unsupported operating system. Please open the file manually.")
    except Exception as e:
        print(f"Error opening file: {e}")

search_and_download_books("History reliability")  # Example query
