import os
import webview
from backend.api import Api

def main():
    api = Api()

    # Locate the frontend directory
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    entry_html = os.path.join(frontend_dir, 'index.html')

    window = webview.create_window(
        'PDF Chunker',
        url=entry_html,
        js_api=api,
        width=1200,
        height=800
    )

    api.set_window(window)
    webview.start(debug=False)

if __name__ == '__main__':
    main()
