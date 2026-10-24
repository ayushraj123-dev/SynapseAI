"""SynapseAI - Futuristic Color & Theme Design System.

Supports vibrant, accessible Dark and Light themes with electric violet, vivid indigo,
electric cyan, bright blue, magenta, and teal accents.
"""


def get_custom_css(theme: str = "dark") -> str:
    """Generate futuristic, accessible custom CSS tailored for the selected theme mode ('dark' or 'light').

    Features electric violet, vivid indigo, electric cyan, magenta, and teal accents with rich spatial depth.
    """
    is_light = (theme == "light")

    if is_light:
        app_bg = "radial-gradient(circle at 15% 15%, #eef2fb 0%, #f4f6fc 60%, #e2e8f0 100%) !important"
        app_text = "#172033"
        sidebar_bg = "#f8fafc !important"
        sidebar_border = "#dce3f2 !important"
        heading_color = "#0f172a !important"
        subtext_color = "#475569"

        brand_hdr_bg = "linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.95) 100%)"
        brand_hdr_border = "rgba(99, 102, 241, 0.25)"
        brand_hdr_shadow = "0 10px 30px -10px rgba(99, 102, 241, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.9)"
        brand_title_color = "#0f172a"
        brand_headline_color = "#6366f1"

        card_bg = "#ffffff !important"
        card_border = "#dce3f2 !important"
        card_text = "#0f172a !important"
        card_shadow = "0 4px 18px rgba(0, 0, 0, 0.05) !important"
        card_hover_border = "rgba(99, 102, 241, 0.4) !important"

        subtle_card_bg = "#f8fafc !important"
        subtle_card_border = "#e2e8f0 !important"
        subtle_card_text = "#0f172a !important"

        stat_badge_bg = "#ffffff"
        stat_badge_border = "#dce3f2"
        stat_badge_text = "#0f172a"

        input_bg = "#ffffff !important"
        input_border = "#cbd5e1 !important"
        input_text = "#0f172a !important"

        metric_val_color = "#0f172a !important"
        citation_box_bg = "#ffffff"
        citation_box_border = "#cbd5e1"
        citation_box_text = "#0f172a"
        passage_card_bg = "#ffffff"
        passage_card_border = "#cbd5e1"
        passage_card_text = "#0f172a"
        grounded_box_bg = "rgba(240, 249, 255, 0.96)"
        grounded_box_border = "rgba(14, 165, 233, 0.45)"
        grounded_box_title = "#0369a1"
        grounded_box_text = "#0f172a"

        stepper_bg = "rgba(241, 245, 249, 0.85)"
        stepper_border = "#dce3f2"
        status_pill_bg = "#f8fafc"
        status_pill_border = "#e2e8f0"
        status_pill_text = "#0f172a"
        legend_chip_bg = "#ffffff"
        legend_chip_border = "#cbd5e1"
        tab_text_color = "#475569"
        tab_active_color = "#4f46e5"
        expander_title_color = "#0f172a"

        # Centralized Button Design Tokens (Light Mode)
        btn_primary_bg = "linear-gradient(120deg, #7c3aed 0%, #4f46e5 50%, #06b6d4 100%)"
        btn_primary_text = "#ffffff"
        btn_primary_border = "none"
        btn_primary_shadow = "0 4px 14px rgba(99, 102, 241, 0.3)"
        btn_primary_hover_bg = "linear-gradient(120deg, #6d28d9 0%, #4338ca 50%, #0891b2 100%)"
        btn_primary_hover_text = "#ffffff"
        btn_primary_hover_shadow = "0 6px 20px rgba(99, 102, 241, 0.45)"

        btn_secondary_bg = "#ffffff"
        btn_secondary_text = "#0f172a"
        btn_secondary_border = "1px solid #cbd5e1"
        btn_secondary_shadow = "0 2px 8px rgba(0, 0, 0, 0.05)"
        btn_secondary_hover_bg = "#f8fafc"
        btn_secondary_hover_text = "#4338ca"
        btn_secondary_hover_border = "#8b5cf6"
        btn_secondary_hover_shadow = "0 4px 14px rgba(99, 102, 241, 0.18)"
        btn_secondary_active_bg = "#f1f5f9"
        btn_secondary_active_text = "#0f172a"

        btn_disabled_bg = "#f1f5f9"
        btn_disabled_text = "#94a3b8"
        btn_disabled_border = "1px solid #e2e8f0"
        btn_focus_ring = "0 0 0 3px rgba(99, 102, 241, 0.35)"
    else:
        app_bg = "radial-gradient(circle at 15% 15%, rgba(139, 92, 246, 0.14) 0%, rgba(11, 16, 32, 0.98) 45%, #060913 100%) !important"
        app_text = "#e2e8f0"
        sidebar_bg = "#0b1020 !important"
        sidebar_border = "rgba(139, 92, 246, 0.18) !important"
        heading_color = "#f8fafc !important"
        subtext_color = "#94a3b8"

        brand_hdr_bg = "linear-gradient(135deg, rgba(17, 24, 43, 0.90) 0%, rgba(11, 16, 32, 0.95) 100%)"
        brand_hdr_border = "rgba(139, 92, 246, 0.25)"
        brand_hdr_shadow = "0 10px 30px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1)"
        brand_title_color = "#f8fafc"
        brand_headline_color = "#22d3ee"

        card_bg = "rgba(17, 24, 43, 0.85) !important"
        card_border = "rgba(139, 92, 246, 0.22) !important"
        card_text = "#f8fafc !important"
        card_shadow = "0 6px 24px rgba(0, 0, 0, 0.35) !important"
        card_hover_border = "rgba(56, 189, 248, 0.35) !important"

        subtle_card_bg = "#11182b !important"
        subtle_card_border = "rgba(99, 102, 241, 0.25) !important"
        subtle_card_text = "#e2e8f0 !important"

        stat_badge_bg = "rgba(15, 23, 42, 0.65)"
        stat_badge_border = "rgba(255, 255, 255, 0.08)"
        stat_badge_text = "#e2e8f0"

        input_bg = "#131b2e !important"
        input_border = "rgba(99, 102, 241, 0.3) !important"
        input_text = "#f3f4f6 !important"

        metric_val_color = "#f8fafc !important"
        citation_box_bg = "rgba(13, 20, 38, 0.85)"
        citation_box_border = "rgba(56, 189, 248, 0.25)"
        citation_box_text = "#f1f5f9"
        passage_card_bg = "rgba(17, 24, 43, 0.85)"
        passage_card_border = "rgba(56, 189, 248, 0.25)"
        passage_card_text = "#f1f5f9"
        grounded_box_bg = "rgba(13, 21, 39, 0.85)"
        grounded_box_border = "rgba(56, 189, 248, 0.25)"
        grounded_box_title = "#38bdf8"
        grounded_box_text = "#f1f5f9"

        stepper_bg = "rgba(15, 23, 42, 0.6)"
        stepper_border = "rgba(255, 255, 255, 0.06)"
        status_pill_bg = "rgba(15, 23, 42, 0.6)"
        status_pill_border = "rgba(255, 255, 255, 0.06)"
        status_pill_text = "#e2e8f0"
        legend_chip_bg = "rgba(15, 23, 42, 0.8)"
        legend_chip_border = "rgba(255, 255, 255, 0.1)"
        tab_text_color = "#94a3b8"
        tab_active_color = "#38bdf8"
        expander_title_color = "#f8fafc"

        # Centralized Button Design Tokens (Dark Mode)
        btn_primary_bg = "linear-gradient(120deg, #8B5CF6 0%, #6366F1 50%, #22D3EE 100%)"
        btn_primary_text = "#ffffff"
        btn_primary_border = "none"
        btn_primary_shadow = "0 4px 18px rgba(99, 102, 241, 0.4)"
        btn_primary_hover_bg = "linear-gradient(120deg, #9d71f7 0%, #7579f8 50%, #38dcf0 100%)"
        btn_primary_hover_text = "#ffffff"
        btn_primary_hover_shadow = "0 6px 26px rgba(139, 92, 246, 0.6)"

        btn_secondary_bg = "linear-gradient(135deg, rgba(20, 28, 48, 0.95) 0%, rgba(13, 20, 38, 0.95) 100%)"
        btn_secondary_text = "#f8fafc"
        btn_secondary_border = "1px solid rgba(139, 92, 246, 0.35)"
        btn_secondary_shadow = "0 4px 16px rgba(0, 0, 0, 0.4)"
        btn_secondary_hover_bg = "linear-gradient(135deg, rgba(30, 42, 74, 0.98) 0%, rgba(20, 28, 55, 0.98) 100%)"
        btn_secondary_hover_text = "#38bdf8"
        btn_secondary_hover_border = "rgba(56, 189, 248, 0.7)"
        btn_secondary_hover_shadow = "0 6px 22px rgba(139, 92, 246, 0.35), 0 0 12px rgba(56, 189, 248, 0.25)"
        btn_secondary_active_bg = "rgba(35, 48, 85, 0.98)"
        btn_secondary_active_text = "#38bdf8"

        btn_disabled_bg = "rgba(15, 23, 42, 0.5)"
        btn_disabled_text = "#64748b"
        btn_disabled_border = "1px solid rgba(255, 255, 255, 0.08)"
        btn_focus_ring = "0 0 0 3px rgba(34, 211, 238, 0.45)"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root brand tokens */
        :root {{
            --brand-violet: #8B5CF6;
            --brand-indigo: #6366F1;
            --brand-cyan: #22D3EE;
            --brand-blue: #3B82F6;
            --brand-magenta: #EC4899;
            --brand-teal: #2DD4BF;
            --brand-amber: #F59E0B;
        }}

        /* Global application background and typography */
        .stApp {{
            background: {app_bg};
            color: {app_text};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {sidebar_border};
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.025em;
            color: {heading_color};
        }}

        code, pre {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Brand Header */
        .brand-header {{
            background: {brand_hdr_bg};
            border: 1px solid {brand_hdr_border};
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: {brand_hdr_shadow};
            position: relative;
            overflow: hidden;
        }}

        .brand-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #8B5CF6 0%, #6366F1 35%, #22D3EE 75%, #EC4899 100%);
        }}

        .brand-title {{
            font-size: 2.1rem;
            font-weight: 900;
            color: {brand_title_color};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            letter-spacing: -0.03em;
        }}

        .brand-headline {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {brand_headline_color};
            margin-top: 0.35rem;
            margin-bottom: 0.3rem;
            letter-spacing: -0.01em;
        }}

        .brand-subtitle {{
            font-size: 0.92rem;
            color: {subtext_color};
            margin: 0;
            line-height: 1.55;
            max-width: 900px;
        }}

        .brand-stats-row {{
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(139, 92, 246, 0.15);
        }}

        .stat-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            border-radius: 9999px;
            background: {stat_badge_bg};
            border: 1px solid {stat_badge_border};
            color: {stat_badge_text};
            transition: all 0.2s ease;
        }}

        .stat-badge:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
        }}

        .stat-badge-cyan {{
            border-color: rgba(34, 211, 238, 0.4);
            color: #06b6d4;
        }}

        .stat-badge-violet {{
            border-color: rgba(139, 92, 246, 0.4);
            color: #8b5cf6;
        }}

        .stat-badge-teal {{
            border-color: rgba(45, 212, 191, 0.4);
            color: #0d9488;
        }}

        /* Panel & Item Cards */
        .panel-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: {card_shadow};
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        }}

        .panel-card:hover {{
            border-color: {card_hover_border};
            transform: translateY(-1px);
        }}

        .subtle-card {{
            background: {subtle_card_bg};
            border: 1px solid {subtle_card_border};
            color: {subtle_card_text};
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.5rem;
            font-size: 0.83rem;
        }}

        .panel-heading {{
            font-size: 1.12rem;
            font-weight: 700;
            color: {heading_color};
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Pipeline Processing Stepper */
        .stepper-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: {stepper_bg};
            border: 1px solid {stepper_border};
            border-radius: 10px;
            margin: 0.75rem 0 1rem 0;
            overflow-x: auto;
        }}

        .stepper-node {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: {subtext_color};
            white-space: nowrap;
        }}

        .stepper-node.done {{
            color: #8b5cf6;
        }}

        .stepper-circle {{
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.25);
        }}

        .stepper-node.done .stepper-circle {{
            background: rgba(139, 92, 246, 0.25);
            border-color: #8b5cf6;
            color: #8b5cf6;
            font-weight: 700;
        }}

        .stepper-arrow {{
            color: rgba(139, 92, 246, 0.35);
            font-size: 0.8rem;
        }}

        /* Status Pills */
        .status-pill {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0.85rem;
            border-radius: 8px;
            background-color: {status_pill_bg};
            border: 1px solid {status_pill_border};
            margin-bottom: 0.45rem;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}

        .status-pill-label {{
            color: {status_pill_text};
            font-weight: 500;
        }}

        .status-pill-tag {{
            font-size: 0.74rem;
            font-weight: 600;
            padding: 0.18rem 0.55rem;
            border-radius: 6px;
        }}

        .tag-standby {{
            background-color: rgba(34, 211, 238, 0.12);
            color: #0891b2;
            border: 1px solid rgba(34, 211, 238, 0.3);
        }}

        .tag-ready {{
            background-color: rgba(45, 212, 191, 0.15);
            color: #0d9488;
            border: 1px solid rgba(45, 212, 191, 0.35);
        }}

        /* Item Cards */
        .item-card {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            color: {card_text};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.85rem;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }}

        .item-card:hover {{
            border-color: {card_hover_border};
            transform: translateY(-1px);
        }}

        .item-card * {{
            color: inherit;
        }}

        .item-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }}

        .type-badge {{
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            background-color: rgba(139, 92, 246, 0.15);
            color: #8b5cf6 !important;
            border: 1px solid rgba(139, 92, 246, 0.35);
        }}

        .pred-badge {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.55rem;
            border-radius: 6px;
            background-color: rgba(245, 158, 11, 0.15);
            color: #d97706 !important;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        /* Document Chunk Monospace Text Previews */
        .chunk-preview-text {{
            font-size: 0.85rem;
            color: {card_text};
            font-family: 'JetBrains Mono', monospace;
            white-space: pre-wrap;
            line-height: 1.55;
            background: {subtle_card_bg};
            border: 1px solid {subtle_card_border};
            border-radius: 6px;
            padding: 0.65rem 0.85rem;
            margin-top: 0.4rem;
        }}

        /* Evidence and Provenance Quotes */
        .evidence-quote {{
            font-size: 0.85rem;
            color: {card_text};
            font-style: italic;
            border-left: 2px solid #8b5cf6;
            padding-left: 0.65rem;
            margin-top: 0.4rem;
            line-height: 1.5;
        }}

        .passage-card {{
            background-color: {passage_card_bg};
            border: 1px solid {passage_card_border};
            border-left: 3px solid #22d3ee;
            color: {passage_card_text};
            border-radius: 10px;
            padding: 0.95rem 1.2rem;
            margin-bottom: 0.85rem;
            transition: all 0.2s ease;
        }}

        .passage-card:hover {{
            border-left-color: #8b5cf6;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.15);
            transform: translateY(-1px);
        }}

        .passage-text {{
            font-size: 0.88rem;
            color: {passage_card_text};
            line-height: 1.55;
            margin: 0.6rem 0;
        }}

        /* Grounded Synthesized Response Card */
        .grounded-response-card {{
            background: {grounded_box_bg};
            border: 1px solid {grounded_box_border};
            border-left: 4px solid #38bdf8;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(12px);
        }}

        .grounded-response-title {{
            font-size: 0.86rem;
            font-weight: 700;
            color: {grounded_box_title};
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .grounded-response-text {{
            font-size: 0.95rem;
            color: {grounded_box_text};
            line-height: 1.65;
        }}

        .score-badge {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.18rem 0.6rem;
            border-radius: 9999px;
            background-color: rgba(34, 211, 238, 0.15);
            color: #0891b2;
            border: 1px solid rgba(34, 211, 238, 0.4);
        }}

        .legend-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.75rem;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            background-color: {legend_chip_bg};
            border: 1px solid {legend_chip_border};
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            transition: all 0.15s ease;
        }}

        .legend-color-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        /* Q&A Badges */
        .intent-badge {{
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            letter-spacing: 0.03em;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .intent-factual {{
            background-color: rgba(34, 211, 238, 0.18);
            color: #0891b2;
            border: 1px solid rgba(34, 211, 238, 0.45);
        }}

        .intent-relationship {{
            background-color: rgba(45, 212, 191, 0.18);
            color: #0d9488;
            border: 1px solid rgba(45, 212, 191, 0.45);
        }}

        .intent-thematic {{
            background-color: rgba(139, 92, 246, 0.18);
            color: #8b5cf6;
            border: 1px solid rgba(139, 92, 246, 0.45);
        }}

        .intent-unanswerable {{
            background-color: rgba(245, 158, 11, 0.18);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.45);
        }}

        .suff-pill {{
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.2rem 0.7rem;
            border-radius: 8px;
        }}

        .suff-supported {{
            background-color: rgba(34, 197, 94, 0.15);
            color: #16a34a;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }}

        .suff-partial {{
            background-color: rgba(234, 179, 8, 0.15);
            color: #ca8a04;
            border: 1px solid rgba(234, 179, 8, 0.35);
        }}

        .suff-insufficient {{
            background-color: rgba(245, 158, 11, 0.15);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        /* Citation Cards */
        .citation-box {{
            background: {citation_box_bg};
            border: 1px solid {citation_box_border};
            border-left: 3px solid #8b5cf6;
            color: {citation_box_text} !important;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.84rem;
            transition: all 0.2s ease;
        }}

        .citation-box:hover {{
            border-color: rgba(139, 92, 246, 0.5);
            transform: translateX(2px);
        }}

        .citation-title {{
            font-weight: 600;
            color: {citation_box_text};
        }}

        .citation-quote {{
            color: {citation_box_text};
            opacity: 0.92;
            font-style: italic;
            border-left: 2px solid #8b5cf6;
            padding-left: 0.55rem;
            margin-top: 0.35rem;
            line-height: 1.5;
        }}

        .citation-tag {{
            font-weight: 700;
            color: #8b5cf6;
            background-color: rgba(139, 92, 246, 0.15);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            margin-right: 0.45rem;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}

        /* Inputs & Controls styling for both themes */
        div[data-baseweb="input"], div[data-baseweb="select"] > div, textarea {{
            background-color: {input_bg};
            color: {input_text};
            border-color: {input_border};
            border-radius: 8px !important;
        }}

        div[data-testid="stMetricValue"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 800 !important;
            color: {metric_val_color};
        }}

        /* =====================================================================
           Centralized High-Contrast Button System (All Themes & States)
           ===================================================================== */
        /* Base button typography and interactive defaults */
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button,
        .stLinkButton > a,
        button[kind="primary"],
        button[kind="secondary"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.90rem !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            outline: none !important;
            text-decoration: none !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        /* 1. Primary Buttons (Gradient with crisp white text) */
        button[kind="primary"],
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        .stLinkButton > a[kind="primary"] {{
            background: {btn_primary_bg} !important;
            border: {btn_primary_border} !important;
            box-shadow: {btn_primary_shadow} !important;
            font-weight: 700 !important;
            color: {btn_primary_text} !important;
        }}

        button[kind="primary"] *,
        .stButton > button[kind="primary"] *,
        .stFormSubmitButton > button[kind="primary"] *,
        .stLinkButton > a[kind="primary"] * {{
            color: {btn_primary_text} !important;
            font-weight: 700 !important;
        }}

        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover,
        .stLinkButton > a[kind="primary"]:hover {{
            background: {btn_primary_hover_bg} !important;
            color: {btn_primary_hover_text} !important;
            box-shadow: {btn_primary_hover_shadow} !important;
            transform: translateY(-1px) !important;
        }}

        button[kind="primary"]:hover *,
        .stButton > button[kind="primary"]:hover *,
        .stFormSubmitButton > button[kind="primary"]:hover *,
        .stLinkButton > a[kind="primary"]:hover * {{
            color: {btn_primary_hover_text} !important;
        }}

        button[kind="primary"]:active,
        .stButton > button[kind="primary"]:active,
        .stFormSubmitButton > button[kind="primary"]:active,
        .stLinkButton > a[kind="primary"]:active {{
            transform: translateY(0) !important;
            box-shadow: {btn_primary_shadow} !important;
            opacity: 0.95 !important;
        }}

        button[kind="primary"]:focus-visible,
        .stButton > button[kind="primary"]:focus-visible,
        .stFormSubmitButton > button[kind="primary"]:focus-visible,
        .stLinkButton > a[kind="primary"]:focus-visible {{
            box-shadow: {btn_focus_ring} !important;
            outline: none !important;
        }}

        /* 2. Secondary & Utility Buttons (Theme-aware, high-contrast, no white-on-white) */
        button[kind="secondary"],
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button,
        .stLinkButton > a:not([kind="primary"]),
        section[data-testid="stSidebar"] .stButton > button {{
            background: {btn_secondary_bg} !important;
            border: {btn_secondary_border} !important;
            box-shadow: {btn_secondary_shadow} !important;
            font-weight: 600 !important;
            color: {btn_secondary_text} !important;
        }}

        button[kind="secondary"] *,
        .stButton > button:not([kind="primary"]) *,
        .stDownloadButton > button *,
        .stLinkButton > a:not([kind="primary"]) *,
        section[data-testid="stSidebar"] .stButton > button * {{
            color: {btn_secondary_text} !important;
            font-weight: 600 !important;
        }}

        /* Secondary Hover State */
        button[kind="secondary"]:hover,
        .stButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover,
        .stLinkButton > a:not([kind="primary"]):hover,
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: {btn_secondary_hover_bg} !important;
            border-color: {btn_secondary_hover_border} !important;
            box-shadow: {btn_secondary_hover_shadow} !important;
            color: {btn_secondary_hover_text} !important;
            transform: translateY(-1px) !important;
        }}

        button[kind="secondary"]:hover *,
        .stButton > button:not([kind="primary"]):hover *,
        .stDownloadButton > button:hover *,
        .stLinkButton > a:not([kind="primary"]):hover *,
        section[data-testid="stSidebar"] .stButton > button:hover * {{
            color: {btn_secondary_hover_text} !important;
        }}

        /* Secondary Active / Pressed State */
        button[kind="secondary"]:active,
        .stButton > button:not([kind="primary"]):active,
        .stDownloadButton > button:active,
        .stLinkButton > a:not([kind="primary"]):active,
        section[data-testid="stSidebar"] .stButton > button:active {{
            background: {btn_secondary_active_bg} !important;
            color: {btn_secondary_active_text} !important;
            transform: translateY(0) !important;
        }}

        button[kind="secondary"]:active *,
        .stButton > button:not([kind="primary"]):active *,
        .stDownloadButton > button:active *,
        .stLinkButton > a:not([kind="primary"]):active *,
        section[data-testid="stSidebar"] .stButton > button:active * {{
            color: {btn_secondary_active_text} !important;
        }}

        /* Secondary Keyboard Focus Ring */
        button[kind="secondary"]:focus-visible,
        .stButton > button:not([kind="primary"]):focus-visible,
        .stDownloadButton > button:focus-visible,
        .stLinkButton > a:not([kind="primary"]):focus-visible,
        section[data-testid="stSidebar"] .stButton > button:focus-visible {{
            box-shadow: {btn_focus_ring} !important;
            outline: none !important;
        }}

        /* 3. Disabled State */
        button:disabled,
        button[disabled],
        .stButton > button:disabled,
        .stDownloadButton > button:disabled,
        .stFormSubmitButton > button:disabled {{
            background: {btn_disabled_bg} !important;
            border: {btn_disabled_border} !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
            transform: none !important;
            opacity: 0.65 !important;
        }}

        button:disabled *,
        button[disabled] *,
        .stButton > button:disabled *,
        .stDownloadButton > button:disabled *,
        .stFormSubmitButton > button:disabled * {{
            color: {btn_disabled_text} !important;
        }}

        /* 4. Safeguard: Any button with an explicit white or near-white background MUST have dark text */
        button[style*="background: white"],
        button[style*="background-color: white"],
        button[style*="background: #fff"],
        button[style*="background-color: #fff"],
        button[style*="background:#ffffff"],
        button[style*="background-color:#ffffff"],
        .white-btn {{
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }}

        button[style*="background: white"] *,
        button[style*="background-color: white"] *,
        button[style*="background: #fff"] *,
        button[style*="background-color: #fff"] *,
        button[style*="background:#ffffff"] *,
        button[style*="background-color:#ffffff"] *,
        .white-btn * {{
            color: #0f172a !important;
        }}

        /* Tabs styling */
        button[data-baseweb="tab"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: {tab_text_color} !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {tab_active_color} !important;
        }}

        /* Expanders styling */
        div[data-testid="stExpander"] details summary span {{
            color: {expander_title_color} !important;
            font-weight: 600 !important;
        }}

        /* Contact Us & Footer Styling */
        .contact-hero-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: {card_shadow};
            position: relative;
            overflow: hidden;
        }}

        .contact-hero-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #8B5CF6 0%, #22D3EE 50%, #EC4899 100%);
        }}

        .contact-email-box {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: {subtle_card_bg};
            border: 1px solid {subtle_card_border};
            border-radius: 10px;
            padding: 0.85rem 1.15rem;
            margin: 0.8rem 0 1.1rem 0;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .contact-email-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.05rem;
            font-weight: 600;
            color: #8b5cf6;
            letter-spacing: -0.01em;
            user-select: all;
            -webkit-user-select: all;
        }}

        .contact-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.82rem;
            color: {subtext_color};
            background: {stat_badge_bg};
            border: 1px solid {stat_badge_border};
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }}

        .app-footer {{
            border-top: 1px solid rgba(139, 92, 246, 0.2);
            padding: 1.25rem 0 0.5rem 0;
            margin-top: 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            color: {subtext_color};
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .app-footer a {{
            color: #8b5cf6 !important;
            text-decoration: none !important;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .app-footer a:hover {{
            color: #22d3ee !important;
            text-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
        }}

        /* Phase 2: Contradictions, Multilingual & Ingestion Telemetry */
        .contradiction-card {{
            background: rgba(245, 158, 11, 0.07);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-left: 5px solid #f59e0b;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(245, 158, 11, 0.1);
        }}

        .contradiction-badge {{
            background: linear-gradient(135deg, #f59e0b 0%, #ec4899 100%);
            color: #ffffff !important;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .multilingual-banner {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(34, 211, 238, 0.12) 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 10px;
            padding: 0.85rem 1.2rem;
            margin: 0.8rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .sla-badge-success {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981 !important;
            border: 1px solid rgba(16, 185, 129, 0.4);
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
        }}

        .benchmark-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: {card_shadow};
        }}

        /* Accessibility: Reduced Motion Support */
        @media (prefers-reduced-motion: reduce) {{
            *, ::before, ::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }}
        }}
    </style>
    """
