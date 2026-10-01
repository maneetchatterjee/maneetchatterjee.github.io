# Maneet Chatterjee — research portfolio

Personal academic and engineering portfolio at **https://maneetchatterjee.github.io/**.

A dependency-free static site for GitHub Pages. No package install, build step, analytics, or backend is required.

## Preview locally

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000` from the repository root.

## Edit the portfolio

- `index.html`: biography, selected projects, papers, experience, education, contact, and metadata.
- `assets/styles.css`: responsive layout, light/dark themes, and print styling.
- `assets/site.js`: theme toggle and active navigation. All content and project details work without JavaScript.
- `assets/theme.js`: applies the saved theme before first paint, with system preference as fallback.
- `assets/Maneet_Chatterjee_CV.pdf`: CV supplied in October 2026. Replace it in place when updating the CV.
- `Images/maneet_profile.jpg`: existing portrait. Research figures remain available in `Images/`.
- `content/*.html`: redirects retain previously shared URLs.

The homepage deliberately lists relevant research rather than an exhaustive CV. CV metrics retain their evaluation context. Publication status is explicit: the MONTI 2026 paper is a non-archival CVPR workshop paper, and BMD-CD is an arXiv preprint. Only confirmed public repositories and specific paper links are linked; the private VLM project has no public code button.

## Content sources

- The supplied `Maneet_CV_one_page.pdf` is authoritative for experience, education, selected projects, honors, and four selected publications.
- [BMD-CD, arXiv:2609.27149](https://arxiv.org/abs/2609.27149) supplies the September 2026 preprint title, authors, equal-contribution note, and code URL.
- [MONTI 2026](https://sites.google.com/view/monti2026/home) supplies the accepted workshop paper link and confirms non-archival status.
- [MoonBot](https://github.com/maneetchatterjee/MoonBot) documents the simulation pipeline and identifies physical deployment as future work.

When updating, verify author order, venue, publication status, external links, and the date in the footer. Do not infer an accepted conference from a repository name.

## Deployment

GitHub Pages serves the repository root from `main` using the existing configuration. Retain that configuration; this redesign needs no new hosting service.
