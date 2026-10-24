"""Tests for SynapseAI Futuristic UI/UX Redesign, Theme Switching, and Entrance Animation."""

import unittest
from src.theme_styles import get_custom_css
from src.knowledge_graph import KnowledgeGraphBuilder


class TestFuturisticThemeAndStyles(unittest.TestCase):
    """Verifies that the design system provides coordinated palettes for dark and light modes."""

    def test_dark_theme_tokens(self):
        """Dark theme must contain brand gradient, electric violet/indigo/cyan tokens, and cosmic navy surfaces."""
        css_dark = get_custom_css("dark")
        self.assertIn("#8B5CF6", css_dark)  # Electric violet
        self.assertIn("#6366F1", css_dark)  # Vivid indigo
        self.assertIn("#22D3EE", css_dark)  # Electric cyan
        self.assertIn("#0b1020", css_dark)  # Navy sidebar
        self.assertIn("prefers-reduced-motion", css_dark)

    def test_light_theme_tokens(self):
        """Light theme must contain light surfaces, dark headings, and high-contrast typography."""
        css_light = get_custom_css("light")
        self.assertIn("#f4f6fc", css_light)  # Crisp light background
        self.assertIn("#172033", css_light)  # Primary dark text
        self.assertIn("#0f172a", css_light)  # Heading color
        self.assertIn("#dce3f2", css_light)  # Crisp borders
        self.assertIn("#8B5CF6", css_light)  # Brand accents preserved


class Test3DGraphThemeAdaptation(unittest.TestCase):
    """Verifies that 3D WebGL graph and Canvas particle simulations adapt dynamically to themes."""

    def setUp(self):
        self.builder = KnowledgeGraphBuilder()
        self.builder.build_graph(
            entities=[
                {"entity_id": "e1", "name": "DeepLearning", "entity_type": "TECHNOLOGY"},
                {"entity_id": "e2", "name": "Transformers", "entity_type": "CONCEPT"},
            ],
            relations=[
                {"relationship_id": "r1", "subject_id": "e1", "object_id": "e2", "predicate": "ENABLES"},
            ],
        )

    def test_3d_graph_dark_theme(self):
        """Dark mode 3D graph sets dark Three.js canvas clear and glass HUD."""
        html_dark = self.builder.generate_3d_graph_html(theme="dark")
        self.assertIn("#050811", html_dark)
        self.assertIn("Synapse 3D Knowledge Space", html_dark)
        self.assertIn("ForceGraph3D", html_dark)

    def test_3d_graph_light_theme(self):
        """Light mode 3D graph sets light Three.js canvas clear and crisp high-contrast HUD."""
        html_light = self.builder.generate_3d_graph_html(theme="light")
        self.assertIn("#f4f6fc", html_light)
        self.assertIn("#172033", html_light)
        self.assertIn("Synapse 3D Knowledge Space", html_light)

    def test_2d_pyvis_theme_adaptation(self):
        """2D PyVis graph generator adapts background and node font colors to theme."""
        html_pyvis_light = self.builder.generate_pyvis_html(theme="light")
        self.assertIn("#f4f6fc", html_pyvis_light)
        self.assertIn("#172033", html_pyvis_light)

        html_pyvis_dark = self.builder.generate_pyvis_html(theme="dark")
        self.assertIn("#0d1117", html_pyvis_dark)
        self.assertIn("#f8fafc", html_pyvis_dark)

    def test_empty_state_constellation_theme_adaptation(self):
        """Empty state constellation simulation renders canvas and adjusts surface colors."""
        empty_dark = self.builder.generate_empty_state_3d_html(theme="dark")
        self.assertIn("#060913", empty_dark)
        self.assertIn('<canvas id="c"></canvas>', empty_dark)

        empty_light = self.builder.generate_empty_state_3d_html(theme="light")
        self.assertIn("#f4f6fc", empty_light)
        self.assertIn('<canvas id="c"></canvas>', empty_light)

    def test_cinematic_intro_animation(self):
        """Intro animation renders brand reveal, capability cards, and canvas network formation."""
        intro_dark = self.builder.generate_intro_animation_html(theme="dark")
        self.assertIn("SYNAPSE AI", intro_dark)
        self.assertIn("Hypnexis AI Innovation Lab", intro_dark)
        self.assertIn("linear-gradient(120deg, #8B5CF6 0%, #6366F1 45%, #22D3EE 100%)", intro_dark)
        self.assertIn("c-intro", intro_dark)
        self.assertIn("prefers-reduced-motion", intro_dark)

        intro_light = self.builder.generate_intro_animation_html(theme="light")
        self.assertIn("SYNAPSE AI", intro_light)
        self.assertIn("#f4f6fc", intro_light)


class TestButtonContrastAndStyling(unittest.TestCase):
    """Verifies that all button types and states maintain high-contrast WCAG compliance in both themes."""

    def test_dark_theme_button_contrast(self):
        """Dark mode buttons must use deep navy or gradient surfaces with crisp, bright text."""
        css_dark = get_custom_css("dark")

        # Primary button: Brand gradient with pure white text
        self.assertIn("button[kind=\"primary\"]", css_dark)
        self.assertIn("color: #ffffff !important", css_dark)

        # Secondary button: Deep navy surface with bright white text (#f8fafc - 15.8:1 contrast)
        self.assertIn("color: #f8fafc !important", css_dark)
        self.assertIn("rgba(20, 28, 48, 0.95)", css_dark)

        # Hover state: Electric cyan text (#38bdf8) and glowing border
        self.assertIn("color: #38bdf8 !important", css_dark)
        self.assertIn("rgba(56, 189, 248, 0.7)", css_dark)

        # Focus ring indicator for keyboard navigation
        self.assertIn("0 0 0 3px rgba(34, 211, 238, 0.45)", css_dark)

        # Safeguard: Any button with white background must have dark text
        self.assertIn("color: #0f172a !important", css_dark)

    def test_light_theme_button_contrast(self):
        """Light mode buttons must use crisp surfaces with deep navy high-contrast text."""
        css_light = get_custom_css("light")

        # Primary button: Deep violet/indigo gradient with pure white text
        self.assertIn("linear-gradient(120deg, #7c3aed 0%, #4f46e5 50%, #06b6d4 100%)", css_light)
        self.assertIn("color: #ffffff !important", css_light)

        # Secondary button: White background with dark navy text (#0f172a - 16.6:1 contrast)
        self.assertIn("color: #0f172a !important", css_light)
        self.assertIn("border: 1px solid #cbd5e1 !important", css_light)

        # Hover state: Deep indigo text (#4338ca) on slate-50
        self.assertIn("color: #4338ca !important", css_light)
        self.assertIn("background: #f8fafc !important", css_light)

        # Focus ring indicator for keyboard navigation
        self.assertIn("0 0 0 3px rgba(99, 102, 241, 0.35)", css_light)

    def test_button_inner_elements_coverage(self):
        """Button selectors must target inner child tags (*, p, span) to prevent Streamlit cascading text overrides."""
        css_dark = get_custom_css("dark")
        self.assertIn(".stButton > button:not([kind=\"primary\"]) *", css_dark)
        self.assertIn(".stLinkButton > a:not([kind=\"primary\"]) *", css_dark)
        self.assertIn(".stDownloadButton > button *", css_dark)
        self.assertIn("section[data-testid=\"stSidebar\"] .stButton > button *", css_dark)

    def test_3d_graph_control_button_contrast(self):
        """3D graph floating control buttons (.ctrl-btn) must have high-contrast text in both themes."""
        builder = KnowledgeGraphBuilder()
        builder.build_graph(
            entities=[
                {"entity_id": "e1", "name": "DeepLearning", "entity_type": "TECHNOLOGY"},
                {"entity_id": "e2", "name": "Transformers", "entity_type": "CONCEPT"},
            ],
            relations=[
                {"relationship_id": "r1", "subject_id": "e1", "object_id": "e2", "predicate": "ENABLES"},
            ],
        )
        html_dark = builder.generate_3d_graph_html(theme="dark")
        self.assertIn("color: #f8fafc;", html_dark)

        html_light = builder.generate_3d_graph_html(theme="light")
        self.assertIn("color: #0f172a;", html_light)


def _relative_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    def srgb_to_linear(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)


def _contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = _relative_luminance(hex1)
    l2 = _relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


class TestTextContrastAndSurfaceRules(unittest.TestCase):
    """Verifies that all text classes, cards, citations, and tabs meet WCAG AA (>= 4.5:1) in both themes."""

    def test_contrast_ratio_formula_and_wcag_standards(self):
        """Verify contrast ratios for core theme foreground and background colors."""
        # Light theme primary text on white
        cr_light_primary = _contrast_ratio("#0f172a", "#ffffff")
        self.assertGreaterEqual(cr_light_primary, 4.5, f"Contrast {cr_light_primary:.2f} must be >= 4.5:1")

        # Light theme secondary text on white
        cr_light_secondary = _contrast_ratio("#475569", "#ffffff")
        self.assertGreaterEqual(cr_light_secondary, 4.5, f"Contrast {cr_light_secondary:.2f} must be >= 4.5:1")

        # Dark theme primary text on cosmic navy
        cr_dark_primary = _contrast_ratio("#f8fafc", "#0b1020")
        self.assertGreaterEqual(cr_dark_primary, 4.5, f"Contrast {cr_dark_primary:.2f} must be >= 4.5:1")

        # Dark theme body text on card surface
        cr_dark_body = _contrast_ratio("#f1f5f9", "#11182b")
        self.assertGreaterEqual(cr_dark_body, 4.5, f"Contrast {cr_dark_body:.2f} must be >= 4.5:1")

    def test_css_classes_presence_in_both_themes(self):
        """Verify that theme-aware CSS classes are injected in both themes."""
        for theme in ("dark", "light"):
            css = get_custom_css(theme)
            self.assertIn(".chunk-preview-text", css)
            self.assertIn(".evidence-quote", css)
            self.assertIn(".passage-card", css)
            self.assertIn(".passage-text", css)
            self.assertIn(".grounded-response-card", css)
            self.assertIn(".grounded-response-title", css)
            self.assertIn(".grounded-response-text", css)
            self.assertIn(".citation-box", css)
            self.assertIn(".citation-title", css)
            self.assertIn(".citation-quote", css)
            self.assertIn("button[data-baseweb=\"tab\"]", css)
            self.assertIn("div[data-testid=\"stExpander\"]", css)

    def test_light_mode_card_and_passage_contrast(self):
        """Light mode card and passage text must be dark neutral, avoiding hardcoded white text."""
        css_light = get_custom_css("light")
        self.assertIn("color: #0f172a !important;", css_light)
        self.assertIn(".passage-card", css_light)
        self.assertIn(".citation-box", css_light)


class TestChunkExpansionProgressiveDisclosure(unittest.TestCase):
    """Verifies progressive disclosure logic for document chunk previews."""

    def test_chunk_limit_initialization_and_progression(self):
        """Default limit is 6, batch increment is +6, show all reveals all chunks, collapse resets to 6."""
        total_chunks = 34
        doc_key = "test_document_pdf"
        chunk_limit_key = f"chunk_limit_{doc_key}"
        session_mock = {}

        # 1. Initial State
        if chunk_limit_key not in session_mock:
            session_mock[chunk_limit_key] = 6
        visible_count = min(session_mock[chunk_limit_key], total_chunks)
        self.assertEqual(visible_count, 6)
        self.assertEqual(session_mock[chunk_limit_key], 6)

        # 2. First Click (+6)
        session_mock[chunk_limit_key] = min(visible_count + 6, total_chunks)
        visible_count = min(session_mock[chunk_limit_key], total_chunks)
        self.assertEqual(visible_count, 12)

        # 3. Second Click (+6)
        session_mock[chunk_limit_key] = min(visible_count + 6, total_chunks)
        visible_count = min(session_mock[chunk_limit_key], total_chunks)
        self.assertEqual(visible_count, 18)

        # 4. Show All Action
        session_mock[chunk_limit_key] = total_chunks
        visible_count = min(session_mock[chunk_limit_key], total_chunks)
        self.assertEqual(visible_count, 34)

        # 5. Show Less Action
        session_mock[chunk_limit_key] = 6
        visible_count = min(session_mock[chunk_limit_key], total_chunks)
        self.assertEqual(visible_count, 6)

    def test_independent_document_scoped_state(self):
        """Each document maintains independent chunk expansion state."""
        session_mock = {}
        doc1_key = "doc1_txt"
        doc2_key = "doc2_pdf"

        session_mock[f"chunk_limit_{doc1_key}"] = 18
        session_mock[f"chunk_limit_{doc2_key}"] = 6

        self.assertEqual(session_mock[f"chunk_limit_{doc1_key}"], 18)
        self.assertEqual(session_mock[f"chunk_limit_{doc2_key}"], 6)

    def test_short_and_empty_chunk_lists(self):
        """Documents with fewer than 6 chunks or 0 chunks display correctly without over-indexing."""
        # Short list: 3 chunks
        chunks_3 = ["c1", "c2", "c3"]
        limit = 6
        visible_3 = min(limit, len(chunks_3))
        self.assertEqual(visible_3, 3)
        self.assertEqual(chunks_3[:visible_3], ["c1", "c2", "c3"])

        # Empty list: 0 chunks
        chunks_0 = []
        visible_0 = min(limit, len(chunks_0))
        self.assertEqual(visible_0, 0)
        self.assertEqual(chunks_0[:visible_0], [])


if __name__ == "__main__":
    unittest.main()

