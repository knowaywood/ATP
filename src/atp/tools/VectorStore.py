from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import Tool
from langchain_text_splitters import MarkdownTextSplitter

from atp.tools.OCR import pdf_to_markdown

load_dotenv()


def get_markdown_files(dir_path: str) -> set[str]:
    """Retrieve markdown file names from a directory."""
    markdown_files: set[str] = set()
    try:
        for entry in os.scandir(dir_path):
            if entry.is_file() and entry.name.endswith(".md"):
                markdown_files.add(entry.name)
    except OSError as exc:
        print(f"Error: {exc}")

    return markdown_files


def add_del_files(data: list[dict], md_dir: str) -> tuple[set[str], set[str]]:
    """Compute markdown files that should be added or deleted."""
    unique_files: set[str] = set()
    try:
        for item in data:
            source_path = item.get("source")
            if not isinstance(source_path, str):
                print(f"Warning: 'source' is not a string in item {item}. Skipping.")
                continue
            unique_files.add(os.path.basename(source_path))
    except Exception as exc:
        print(f"Error processing metadata list: {exc}")

    current_md_files = get_markdown_files(md_dir)
    add_files = current_md_files - unique_files
    del_files = {f for f in (unique_files - current_md_files) if f.endswith(".md")}
    return add_files, del_files


def extract_pdf_dir_to_markdown(pdf_dir: str, md_dir: str) -> list[str]:
    """Convert PDFs in `pdf_dir` to Markdown files in `md_dir`."""
    source_dir = Path(pdf_dir)
    target_dir = Path(md_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        output_path = target_dir / f"{pdf_path.stem}.md"
        if output_path.exists():
            continue
        pdf_to_markdown(str(pdf_path), str(output_path))
        created.append(output_path.name)
    return created


class StandVectorStore:
    def __init__(self, name: str = "stand"):
        self.embedding_model = DashScopeEmbeddings(model="text-embedding-v3")
        self.vec_store = Chroma(name, self.embedding_model)
        self.retriever = self.vec_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10},
        )
        self.retriever_tool = Tool(
            name="retriever_tool",
            func=self._retriever_tool_func,
            description="This tool searches and returns the information of math data.",
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Add texts to the vector store."""
        self.vec_store.add_texts(texts, metadatas, ids, **kwargs)

    def _retriever_tool_func(self, quary: str) -> str:
        """Search and return information from the vector store."""
        docs = self.retriever.invoke(quary)
        if not docs:
            return "I found no relevant information in the MathData"
        results = []
        for index, doc in enumerate(docs, start=1):
            results.append(f"Document {index} :\n {doc.page_content}")
        return "\n\n".join(results)


class VectorStore(StandVectorStore):
    def __init__(self, md_dir: str, store_dir: str, name: str = "mathdata"):
        super().__init__()
        self.md_dir = md_dir
        self.store_dir = store_dir
        self.vec_store = Chroma(name, self.embedding_model, store_dir)
        self.sync_store()

    def sync_store(self) -> None:
        """Synchronize markdown documents into the vector store."""
        all_metadata = self.vec_store.get(include=["metadatas"])
        metadata_docs = all_metadata["metadatas"]
        adds, dels = add_del_files(metadata_docs, self.md_dir)

        if adds:
            loader = DirectoryLoader(
                self.md_dir,
                glob=list(adds),
                loader_cls=UnstructuredMarkdownLoader,
                use_multithreading=True,
            )
            doc_pages = loader.load()
            text_splitter = MarkdownTextSplitter(chunk_size=150, chunk_overlap=20)
            doc_chunk = text_splitter.split_documents(doc_pages)
            self.vec_store.add_documents(doc_chunk)
            print(f"vec_store 已添加: {adds}\n")

        if dels:
            matching_ids = [
                current_id
                for current_id, current_meta in zip(
                    all_metadata["ids"],
                    all_metadata["metadatas"],
                )
                if os.path.basename(current_meta["source"]) in dels
            ]
            self.vec_store.delete(ids=matching_ids)
            print(f"vec_store 已删除: {dels}\n")

        print("sync_store 完成")

    def sync_md(self, pdf_dir: str) -> list[str]:
        """Convert new PDFs into markdown, then synchronize the vector store."""
        created = extract_pdf_dir_to_markdown(pdf_dir, self.md_dir)
        if not created:
            print("no new pdf files to add")
            return []
        self.sync_store()
        return created
