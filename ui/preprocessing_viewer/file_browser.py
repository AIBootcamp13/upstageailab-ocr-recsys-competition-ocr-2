"""
File Browser Component for Streamlit Preprocessing Viewer.

Provides directory navigation, image filtering, batch loading capabilities,
recent files history, and drag-and-drop support for directories.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image


class FileBrowser:
    """Interactive file browser for preprocessing viewer with advanced navigation features."""

    def __init__(self, base_directory: str | None = None, supported_extensions: list[str] | None = None):
        """
        Initialize file browser.

        Args:
            base_directory: Root directory for browsing (defaults to current working directory)
            supported_extensions: List of supported file extensions (defaults to image formats)
        """
        self.base_directory = Path(base_directory or os.getcwd())
        self.supported_extensions = supported_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        self.history_file = self.base_directory / ".viewer_history.json"
        self.bookmarks_file = self.base_directory / ".viewer_bookmarks.json"

        # Initialize history and bookmarks
        self._load_history()
        self._load_bookmarks()

    def _load_history(self) -> None:
        """Load recent files history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self.recent_files = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.recent_files = []
        else:
            self.recent_files = []

    def _save_history(self) -> None:
        """Save recent files history to disk."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.recent_files, f, indent=2)
        except OSError:
            pass  # Silently fail if we can't save history

    def _load_bookmarks(self) -> None:
        """Load bookmarks from disk."""
        if self.bookmarks_file.exists():
            try:
                with open(self.bookmarks_file) as f:
                    self.bookmarks = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.bookmarks = {}
        else:
            self.bookmarks = {}

    def _save_bookmarks(self) -> None:
        """Save bookmarks to disk."""
        try:
            with open(self.bookmarks_file, "w") as f:
                json.dump(self.bookmarks, f, indent=2)
        except OSError:
            pass

    def add_to_history(self, file_path: str) -> None:
        """Add file to recent history."""
        file_path = str(Path(file_path).resolve())

        # Remove if already exists
        self.recent_files = [f for f in self.recent_files if f["path"] != file_path]

        # Add to beginning
        self.recent_files.insert(0, {"path": file_path, "timestamp": datetime.now().isoformat(), "name": Path(file_path).name})

        # Keep only last 50 files
        self.recent_files = self.recent_files[:50]
        self._save_history()

    def add_bookmark(self, name: str, path: str) -> None:
        """Add a directory bookmark."""
        self.bookmarks[name] = str(Path(path).resolve())
        self._save_bookmarks()

    def remove_bookmark(self, name: str) -> None:
        """Remove a bookmark."""
        if name in self.bookmarks:
            del self.bookmarks[name]
            self._save_bookmarks()

    def get_directory_contents(self, directory: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get contents of directory, separated into files and subdirectories.

        Returns:
            Tuple of (subdirectories, files) where each item is a dict with metadata
        """
        path = Path(directory)

        if not path.exists() or not path.is_dir():
            return [], []

        subdirs = []
        files = []

        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith("."):
                    continue  # Skip hidden files

                stat = item.stat()
                item_info = {
                    "name": item.name,
                    "path": str(item),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "is_dir": item.is_dir(),
                }

                if item.is_dir():
                    # Count files in directory
                    try:
                        file_count = len([f for f in item.iterdir() if f.is_file()])
                        item_info["file_count"] = file_count
                    except PermissionError:
                        item_info["file_count"] = 0
                    subdirs.append(item_info)
                elif item.suffix.lower() in self.supported_extensions:
                    # Get image dimensions if possible
                    try:
                        with Image.open(item) as img:
                            item_info["dimensions"] = img.size
                    except Exception:
                        item_info["dimensions"] = None
                    files.append(item_info)

        except PermissionError:
            return [], []

        return subdirs, files

    def filter_images(self, files: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Filter image files based on criteria.

        Args:
            files: List of file info dictionaries
            filters: Dictionary of filter criteria

        Returns:
            Filtered list of files
        """
        filtered = files.copy()

        # Size filter
        if filters.get("min_size"):
            min_size = filters["min_size"] * 1024  # Convert KB to bytes
            filtered = [f for f in filtered if f["size"] >= min_size]

        if filters.get("max_size"):
            max_size = filters["max_size"] * 1024  # Convert KB to bytes
            filtered = [f for f in filtered if f["size"] <= max_size]

        # Date filter
        if filters.get("days_old"):
            cutoff = datetime.now() - timedelta(days=filters["days_old"])
            filtered = [f for f in filtered if f["modified"] >= cutoff]

        # Name pattern filter
        if filters.get("name_pattern"):
            pattern = filters["name_pattern"].lower()
            filtered = [f for f in filtered if pattern in f["name"].lower()]

        # Sort options
        sort_by = filters.get("sort_by", "name")
        reverse = filters.get("sort_reverse", False)

        if sort_by == "name":
            filtered.sort(key=lambda x: x["name"].lower(), reverse=reverse)
        elif sort_by == "size":
            filtered.sort(key=lambda x: x["size"], reverse=reverse)
        elif sort_by == "modified":
            filtered.sort(key=lambda x: x["modified"], reverse=reverse)

        return filtered

    def render_file_browser(self, key_prefix: str = "file_browser") -> str | None:
        """
        Render the file browser UI component.

        Args:
            key_prefix: Prefix for streamlit widget keys to avoid conflicts

        Returns:
            Selected file path or None
        """
        st.subheader("📁 File Browser")

        # Initialize session state
        if f"{key_prefix}_current_dir" not in st.session_state:
            st.session_state[f"{key_prefix}_current_dir"] = str(self.base_directory)

        if f"{key_prefix}_selected_file" not in st.session_state:
            st.session_state[f"{key_prefix}_selected_file"] = None

        # Navigation controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Directory path input
            current_dir = st.text_input(
                "Current Directory", value=st.session_state[f"{key_prefix}_current_dir"], key=f"{key_prefix}_dir_input"
            )

            if current_dir != st.session_state[f"{key_prefix}_current_dir"]:
                if current_dir and Path(current_dir).exists() and Path(current_dir).is_dir():
                    st.session_state[f"{key_prefix}_current_dir"] = current_dir
                    st.rerun()
                else:
                    st.error("Directory does not exist")

        with col2:
            if st.button("⬆️ Up", key=f"{key_prefix}_up"):
                parent = Path(st.session_state[f"{key_prefix}_current_dir"]).parent
                if parent != Path(st.session_state[f"{key_prefix}_current_dir"]):
                    st.session_state[f"{key_prefix}_current_dir"] = str(parent)
                    st.rerun()

        with col3:
            if st.button("🏠 Home", key=f"{key_prefix}_home"):
                st.session_state[f"{key_prefix}_current_dir"] = str(self.base_directory)
                st.rerun()

        # Bookmarks
        if self.bookmarks:
            st.subheader("📖 Bookmarks")
            cols = st.columns(min(len(self.bookmarks), 4))
            for i, (name, path) in enumerate(self.bookmarks.items()):
                with cols[i % len(cols)]:
                    if st.button(f"📁 {name}", key=f"{key_prefix}_bookmark_{name}"):
                        st.session_state[f"{key_prefix}_current_dir"] = path
                        st.rerun()

        # Get directory contents
        current_path = Path(st.session_state[f"{key_prefix}_current_dir"])
        subdirs, files = self.get_directory_contents(str(current_path))

        # Filters
        with st.expander("🔍 Filters & Sorting", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Size Filters")
                min_size = st.number_input("Min size (KB)", min_value=0, key=f"{key_prefix}_min_size")
                max_size = st.number_input("Max size (KB)", min_value=0, key=f"{key_prefix}_max_size")

            with col2:
                st.subheader("Other Filters")
                days_old = st.number_input("Modified within (days)", min_value=0, key=f"{key_prefix}_days_old")
                name_pattern = st.text_input("Name contains", key=f"{key_prefix}_name_pattern")

            st.subheader("Sorting")
            sort_options = ["name", "size", "modified"]
            sort_by = st.selectbox("Sort by", sort_options, key=f"{key_prefix}_sort_by")
            sort_reverse = st.checkbox("Reverse order", key=f"{key_prefix}_sort_reverse")

            filters = {
                "min_size": min_size if min_size > 0 else None,
                "max_size": max_size if max_size > 0 else None,
                "days_old": days_old if days_old > 0 else None,
                "name_pattern": name_pattern if name_pattern else None,
                "sort_by": sort_by,
                "sort_reverse": sort_reverse,
            }

        # Apply filters
        filtered_files = self.filter_images(files, filters)

        # Display subdirectories
        if subdirs:
            st.subheader("📂 Subdirectories")
            cols = st.columns(min(len(subdirs), 4))
            for i, subdir in enumerate(subdirs):
                with cols[i % len(cols)]:
                    file_count = subdir.get("file_count", 0)
                    if st.button(f"📁 {subdir['name']}\n({file_count} files)", key=f"{key_prefix}_dir_{subdir['name']}"):
                        st.session_state[f"{key_prefix}_current_dir"] = subdir["path"]
                        st.rerun()

        # Display files
        if filtered_files:
            st.subheader(f"🖼️ Images ({len(filtered_files)} found)")

            # File selection
            file_names = [f["name"] for f in filtered_files]
            selected_name = st.selectbox("Select image", [""] + file_names, key=f"{key_prefix}_file_select")

            if selected_name:
                selected_file = next(f for f in filtered_files if f["name"] == selected_name)
                st.session_state[f"{key_prefix}_selected_file"] = selected_file["path"]

                # Display file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", f"{selected_file['size'] / 1024:.1f} KB")
                with col2:
                    st.metric("Modified", selected_file["modified"].strftime("%Y-%m-%d %H:%M"))
                with col3:
                    if selected_file.get("dimensions"):
                        w, h = selected_file["dimensions"]
                        st.metric("Dimensions", f"{w}×{h}")

                # Preview
                try:
                    image = Image.open(selected_file["path"])
                    st.image(image, caption=selected_name, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {e}")

                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Select This Image", key=f"{key_prefix}_select"):
                        self.add_to_history(selected_file["path"])
                        return selected_file["path"]

                with col2:
                    if st.button("🔖 Add to Bookmarks", key=f"{key_prefix}_add_bookmark"):
                        bookmark_name = st.text_input(
                            "Bookmark name", value=Path(selected_file["path"]).parent.name, key=f"{key_prefix}_bookmark_name"
                        )
                        if bookmark_name:
                            self.add_bookmark(bookmark_name, str(Path(selected_file["path"]).parent))
                            st.success(f"Added bookmark '{bookmark_name}'")

        # Recent files
        if self.recent_files:
            with st.expander("🕒 Recent Files", expanded=False):
                for recent in self.recent_files[:10]:  # Show last 10
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(f"{recent['name']}", key=f"{key_prefix}_recent_{recent['path']}"):
                            st.session_state[f"{key_prefix}_current_dir"] = str(Path(recent["path"]).parent)
                            st.session_state[f"{key_prefix}_selected_file"] = recent["path"]
                            st.rerun()
                    with col2:
                        timestamp = datetime.fromisoformat(recent["timestamp"])
                        st.caption(timestamp.strftime("%m/%d %H:%M"))

        return None

    def render_batch_loader(self, key_prefix: str = "batch_loader") -> list[str]:
        """
        Render batch file loader for processing multiple images.

        Returns:
            List of selected file paths
        """
        st.subheader("📦 Batch Image Loader")

        # Initialize session state
        if f"{key_prefix}_selected_files" not in st.session_state:
            st.session_state[f"{key_prefix}_selected_files"] = []

        current_dir = st.session_state.get("file_browser_current_dir", str(self.base_directory))
        _, files = self.get_directory_contents(current_dir)

        if not files:
            st.info("No images found in current directory")
            return []

        # Multi-select
        file_names = [f["name"] for f in files]
        selected_names = st.multiselect("Select images for batch processing", file_names, key=f"{key_prefix}_multiselect")

        selected_files = [f["path"] for f in files if f["name"] in selected_names]
        st.session_state[f"{key_prefix}_selected_files"] = selected_files

        if selected_files:
            st.success(f"Selected {len(selected_files)} images")

            # Show preview grid
            if len(selected_files) <= 9:  # Limit preview to avoid performance issues
                cols = st.columns(min(len(selected_files), 3))
                for i, file_path in enumerate(selected_files):
                    with cols[i % len(cols)]:
                        try:
                            image = Image.open(file_path)
                            st.image(image, caption=Path(file_path).name, width=150)
                        except Exception:
                            st.error(f"Could not load {Path(file_path).name}")

        return selected_files
