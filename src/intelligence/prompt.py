"""
Cricket Intelligence Engine - Gemini Prompts
Expert-crafted prompts for extracting ball-level cricket intelligence from video.
"""


SYSTEM_PROMPT = """You are an elite cricket analyst with 20+ years of experience watching and 
analyzing professional cricket. You have extensive knowledge of bowling techniques, batting 
strokes, pitch conditions, and match situations across all formats (T20, ODI, Test).

Your task is to watch video clips of individual ball deliveries and extract precise, structured 
cricket intelligence. You must analyze every aspect of the delivery — from the bowler's action 
to the batsman's response to the final outcome.

IMPORTANT GUIDELINES:
1. Be CONSERVATIVE with your classifications. If you're not confident, use "unknown".
2. Provide confidence scores (0.0 to 1.0) for each key field.
3. Use the exact enum values specified in the schema — never invent new categories.
4. Your raw_description should be a concise 1-2 sentence summary a coach would write.
5. Pay attention to:
   - The bowler's arm action (over-arm pace vs finger/wrist spin)
   - Where the ball pitches relative to the stumps
   - The ball's trajectory after bouncing (seam, swing, turn)
   - The batsman's foot movement and stroke selection
   - Whether the ball hits bat cleanly, edges, or is missed
   - The final result (runs scored, dot ball, wicket)
"""


EXTRACTION_PROMPT = """Watch this video clip of a single cricket ball delivery carefully.

Analyze EVERY aspect of this delivery and extract structured data.

Focus on:
1. **Bowler Type**: Is this a pace bowler (fast/medium) or spinner? Look at the bowling action 
   and arm speed.
2. **Line**: Where is the ball directed relative to the stumps? 
   - outside_off: wide of off stump
   - off_stump: on or near off stump  
   - middle: targeting middle stump
   - leg: on or near leg stump
   - outside_leg: wide of leg stump
3. **Length**: How far up the pitch does the ball bounce?
   - yorker: at the batsman's feet
   - full: between the popping crease and a good length
   - good: the ideal length, making the batsman uncertain
   - short_of_length: slightly back of a good length
   - short: bouncing well before halfway
4. **Shot Type**: What stroke did the batsman play?
5. **Contact Quality**: Did the bat hit the ball cleanly, edge it, miss, or mistime?
6. **Outcome**: What was the result — runs scored, dot ball, or wicket?
7. **Ball Movement**: Did the ball seam, swing, or turn after pitching?
8. **Bounce Behavior**: Did the ball keep low, bounce normally, or rear up steeply?

If you can identify the players from jersey numbers, graphics, or commentary, include their names.

Return your analysis as structured JSON matching the required schema.
Be conservative — use "unknown" for anything you're less than 60% sure about.
"""


BATCH_EXTRACTION_PROMPT = """You are watching a sequence of ball deliveries from a cricket match.
For each delivery visible in this clip, extract the structured analysis.

Return a JSON array where each element represents one ball delivery.
If you can identify the over number and ball number, include them.

Use the same analysis criteria as for single ball extraction:
- Bowler type, line, length, variation
- Shot type, footwork, contact quality
- Outcome, ball movement, bounce behavior
- Confidence scores for each field

Be conservative with classifications. Use "unknown" when uncertain.
"""


# ── CV-Augmented Prompt ───────────────────────────────────────────────────────

CV_AUGMENTED_TEMPLATE = """Watch this video clip of a single cricket ball delivery carefully.

COMPUTER VISION PRE-ANALYSIS (from object detection models — treat as reliable facts):
{cv_facts_block}

Using the above CV data as grounding context, your job is to determine the things CV cannot:
- **Bowler type** (pace/spin) — look at bowling action and arm speed
- **Shot type** — what stroke did the batsman play?
- **Footwork** — front foot / back foot movement
- **Contact quality** — clean hit, edge, miss, mistimed?
- **Ball movement** — seam, swing, turn after pitching?
- **Bounce behavior** — low, normal, steep?
- **Outcome** — runs scored / wicket / dot?
- **Variation** — slower ball, cutter, bouncer, spin variation?
- **Player names** — if visible from jersey/scoreboard

IMPORTANT:
- For line and length: If CV computed values are present, use them as your PRIMARY answer
  and set confidence ≥ 0.85 for those fields. Only override if you clearly see a contradiction.
- For everything else, use the video as your primary source.
- Still return "unknown" if you're genuinely unsure about non-CV fields.

Return your analysis as structured JSON matching the required schema.
"""


def _format_cv_facts(cv_data: dict) -> str:
    """
    Format the DualModelDetector result into a readable facts block for the prompt.

    Args:
        cv_data: Output from DualModelDetector.analyze_frame() or a summarized dict.
                 Expected keys: geometry, scene
    """
    lines = []

    geometry = cv_data.get("geometry")
    if geometry:
        lines.append(f"  • Line of delivery:  {geometry['line']}  "
                     f"(normalized offset from stumps: {geometry['normalized_x']:+.2f} stump-widths)")
        lines.append(f"  • Length of delivery: {geometry['length']}  "
                     f"(normalized offset from crease: {geometry['normalized_y']:+.2f} stump-heights)")
        ball_px = geometry.get("ball_px")
        if ball_px:
            lines.append(f"  • Ball position in frame: ({ball_px[0]:.0f}, {ball_px[1]:.0f}) px")
        stump_center = geometry.get("stumps_center")
        if stump_center:
            lines.append(f"  • Stumps center: ({stump_center[0]:.0f}, {stump_center[1]:.0f}) px")
    else:
        lines.append("  • Line / Length: could not compute (stumps or ball not detected by CV)")

    scene = cv_data.get("scene", {})
    detected_roles = []
    for role in ("bowler", "batsman", "wicketkeeper", "nonstriker"):
        obj = scene.get(role)
        if obj:
            detected_roles.append(f"{role} (conf={obj['confidence']:.2f})")
    if detected_roles:
        lines.append(f"  • Players detected:   {', '.join(detected_roles)}")

    umpires = scene.get("umpires", [])
    if umpires:
        lines.append(f"  • Umpires on field:   {len(umpires)} detected")

    stumps = scene.get("stumps")
    if stumps:
        lines.append(f"  • Stumps detected:    yes (conf={stumps['confidence']:.2f})")

    ball = scene.get("ball")
    if ball:
        lines.append(f"  • Ball detected:      yes (conf={ball['confidence']:.2f})")
    else:
        lines.append("  • Ball detected:      no — may be occluded or not in frame")

    return "\n".join(lines)


def get_single_ball_prompt() -> str:
    """Get the full prompt for analyzing a single ball delivery (no CV context)."""
    return EXTRACTION_PROMPT


def get_cv_augmented_prompt(cv_data: dict) -> str:
    """
    Get the Gemini prompt pre-loaded with Roboflow CV detection results.

    Args:
        cv_data: Output from DualModelDetector.analyze_frame():
                 { 'geometry': {...}, 'scene': {...}, ... }
                 OR a simplified dict with the same keys.

    Returns:
        Prompt string with CV facts injected as grounded context.
    """
    facts_block = _format_cv_facts(cv_data)
    return CV_AUGMENTED_TEMPLATE.format(cv_facts_block=facts_block)


def get_batch_prompt() -> str:
    """Get the prompt for analyzing multiple deliveries in a clip."""
    return BATCH_EXTRACTION_PROMPT


def get_system_prompt() -> str:
    """Get the system-level instruction prompt."""
    return SYSTEM_PROMPT
