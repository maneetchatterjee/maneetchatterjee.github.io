# Maneet Chatterjee — research portfolio

Personal academic and engineering portfolio at **https://maneetchatterjee.github.io/**.

A dependency-free static site for GitHub Pages. No package install, build step, analytics, or backend is required.

## Preview locally

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000` from the repository root.

## Edit the portfolio

- `index.html`: research overview, recent news, selected projects, publications, experience, education, public profiles, and metadata.
- `assets/styles.css`: responsive layout, light/dark themes, and print styling.
- `assets/site.js`: theme toggle and active navigation. All content and project details work without JavaScript.
- `assets/theme.js`: applies the saved theme before first paint, with system preference as fallback.
- `Images/`: research visuals only; the personal portrait is no longer hosted.
- `content/*.html`: redirects retain previously shared URLs.

The homepage focuses on public research milestones, project outputs, and professional profiles. It intentionally omits email, phone number, personal location, portrait, grade, certificates, and an unredacted CV. Two WACV 2027 main-conference papers are labeled accepted; the CVPR MONTI 2026 paper is described as a workshop paper. Simulation results retain their evaluation context.

## Content sources

- The November 2026 CV informed the research and publications updates. It is not hosted publicly because it contains personal contact information.
- [BMD-CD](https://arxiv.org/abs/2609.27149) and [INTR++](https://arxiv.org/abs/2610.37178) are publicly linked WACV 2027 works.
- [MONTI 2026](https://sites.google.com/view/monti2026/home) is identified as a CVPR workshop.
- [MoonBot](https://github.com/maneetchatterjee/MoonBot) and [SafeVLA-M](https://github.com/maneetchatterjee/SafeVLA-M) are linked public robotics research repositories.
- The news timeline restores the feature of the earlier portfolio and updates it with recent professional milestones.

When updating, verify author order, venue, acceptance status, external links, and milestone dates. Avoid adding personal contact details, location, face photos, or an unredacted CV.

## Deployment

GitHub Pages serves the repository root from `main` using the existing configuration. Retain that configuration; this redesign needs no new hosting service.
