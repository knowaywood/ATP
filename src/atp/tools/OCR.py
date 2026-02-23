from pathlib import Path

from mineru_kie_sdk import MineruKIEClient

client = MineruKIEClient(
    base_url="https://mineru.net/api/kie", pipeline_id="", timeout=30
)


def post_pdf(pdf_path):

    p = Path(pdf_path)
    files = {"file": (p.name, open(p, "rb"), "application/pdf")}


def get_pdf_url():
    pass


def download_zip():
    pass


def unzip_files(path, target_path):
    pass


def move_delete_files():
    pass
