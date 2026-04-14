"""
Cricket Intelligence Engine - Human Review UI
Streamlit app for reviewing and correcting ball-level extractions.
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import CricketDB
from src.intelligence.schema import (
    Line, Length, ShotType, BowlerType,
    Footwork, ContactQuality, Outcome,
    Variation, BounceBehavior, Movement,
)


# ===== Page Config =====
st.set_page_config(
    page_title="Cricket Intelligence - Review",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Custom CSS =====
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }   
    .confidence-high { color: #00ff88; font-weight: bold; }
    .confidence-mid { color: #ffaa00; font-weight: bold; }
    .confidence-low { color: #ff4444; font-weight: bold; }
    .ball-header { 
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem; border-radius: 0.5rem; 
        border: 1px solid #333; margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ===== Initialize DB =====
@st.cache_resource
def get_db():
    return CricketDB()


db = get_db()


# ===== Sidebar =====
st.sidebar.title("🏏 Cricket Intelligence")
st.sidebar.markdown("### Ball Review Dashboard")

# Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["📊 Dashboard", "🔍 Review Balls", "📋 Full Dataset"],
    index=0,
)

# Match filter
matches = db.list_matches()
match_ids = [m.match_id for m in matches]
selected_match = st.sidebar.selectbox(
    "Select Match",
    ["All"] + match_ids,
    index=0,
)
match_filter = None if selected_match == "All" else selected_match


# ===== Dashboard Mode =====
if mode == "📊 Dashboard":
    st.title("📊 Cricket Intelligence Dashboard")

    stats = db.get_stats(match_filter)

    if stats.get("total", 0) == 0:
        st.info(
            "No ball records yet. Run the extraction pipeline first:\n\n"
            "```bash\n"
            "python -m src.intelligence.extractor --dir data/ball_clips/ --match-id my_match\n"
            "```"
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Balls", stats["total"])
        with col2:
            st.metric("Reviewed", f"{stats['reviewed']}/{stats['total']}")
        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        with col4:
            st.metric("Unknowns", stats.get("unknown_count", 0))

        st.markdown("---")

        # Outcome distribution
        if stats.get("outcomes"):
            st.subheader("Outcome Distribution")
            outcomes_df = pd.DataFrame(
                list(stats["outcomes"].items()),
                columns=["Outcome", "Count"],
            )
            st.bar_chart(outcomes_df.set_index("Outcome"))

        # Review progress
        st.subheader("Review Progress")
        review_pct = stats.get("review_pct", 0)
        st.progress(review_pct / 100)
        st.caption(f"{review_pct:.1f}% reviewed")


# ===== Review Mode =====
elif mode == "🔍 Review Balls":
    st.title("🔍 Ball Review Interface")

    # Get unreviewed balls
    review_only = st.checkbox("Show only unreviewed", value=True)

    if review_only:
        balls = db.get_balls_needing_review(match_filter)
    elif match_filter:
        balls = db.get_balls_for_match(match_filter)
    else:
        balls = []

    if not balls:
        st.success("🎉 All balls reviewed!" if review_only else "No balls found.")
    else:
        st.info(f"📝 {len(balls)} balls to review")

        # Ball selector
        ball_options = [f"Over {b.over_number}.{b.ball_number} — {b.ball_id}" for b in balls]
        selected_idx = st.selectbox("Select Ball", range(len(ball_options)), format_func=lambda i: ball_options[i])

        ball = balls[selected_idx]

        st.markdown("---")

        # Two-column layout: Video + Fields
        col_video, col_fields = st.columns([1, 1])

        with col_video:
            st.subheader(f"🎬 Over {ball.over_number}.{ball.ball_number}")

            # Show video clip if available
            if ball.clip_path and Path(ball.clip_path).exists():
                st.video(ball.clip_path)
            else:
                st.warning("Video clip not available")

            # Raw description
            if ball.raw_description:
                st.markdown("**AI Description:**")
                st.info(ball.raw_description)

            # Confidence scores
            st.markdown("**Confidence Scores:**")
            conf_data = {
                "Line": ball.confidence_line,
                "Length": ball.confidence_length,
                "Shot": ball.confidence_shot_type,
                "Outcome": ball.confidence_outcome,
            }
            for field, score in conf_data.items():
                color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                st.write(f"{color} {field}: {score:.0%}")

        with col_fields:
            st.subheader("📝 Edit Fields")

            with st.form(key=f"review_{ball.ball_id}"):
                enum_options = lambda e: [v.value for v in e]

                new_bowler_type = st.selectbox("Bowler Type", enum_options(BowlerType), index=enum_options(BowlerType).index(ball.bowler_type))
                new_line = st.selectbox("Line", enum_options(Line), index=enum_options(Line).index(ball.line))
                new_length = st.selectbox("Length", enum_options(Length), index=enum_options(Length).index(ball.length))
                new_shot = st.selectbox("Shot Type", enum_options(ShotType), index=enum_options(ShotType).index(ball.shot_type))
                new_contact = st.selectbox("Contact", enum_options(ContactQuality), index=enum_options(ContactQuality).index(ball.contact_quality))
                new_outcome = st.selectbox("Outcome", enum_options(Outcome), index=enum_options(Outcome).index(ball.outcome))
                new_footwork = st.selectbox("Footwork", enum_options(Footwork), index=enum_options(Footwork).index(ball.footwork))
                review_notes = st.text_area("Notes", value=ball.review_notes or "")

                submitted = st.form_submit_button("✅ Save Review", use_container_width=True)

                if submitted:
                    updates = {
                        "bowler_type": new_bowler_type,
                        "line": new_line,
                        "length": new_length,
                        "shot_type": new_shot,
                        "contact_quality": new_contact,
                        "outcome": new_outcome,
                        "footwork": new_footwork,
                        "review_notes": review_notes,
                    }
                    db.update_ball_review(ball.ball_id, updates)
                    st.success(f"✅ Ball {ball.ball_id} reviewed!")
                    st.rerun()


# ===== Full Dataset Mode =====
elif mode == "📋 Full Dataset":
    st.title("📋 Full Ball Dataset")

    if match_filter:
        balls = db.get_balls_for_match(match_filter)
    else:
        balls = []
        for m in matches:
            balls.extend(db.get_balls_for_match(m.match_id))

    if not balls:
        st.info("No ball records found.")
    else:
        df = pd.DataFrame([{
            "Ball ID": b.ball_id,
            "Over": f"{b.over_number}.{b.ball_number}",
            "Bowler": b.bowler_type,
            "Line": b.line,
            "Length": b.length,
            "Shot": b.shot_type,
            "Contact": b.contact_quality,
            "Outcome": b.outcome,
            "Confidence": round((b.confidence_line + b.confidence_length + b.confidence_shot_type) / 3, 2),
            "Reviewed": "✅" if b.is_reviewed else "❌",
        } for b in balls])

        st.dataframe(df, use_container_width=True, height=600)

        # Export
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "cricket_intelligence.csv", "text/csv")
