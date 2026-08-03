# Hasan Goni — Data Science & ML Blog

Source for my technical blog, built with [Quarto](https://quarto.org) and published at
**[hasangoni.quarto.pub/hasan-blog-post](https://hasangoni.quarto.pub/hasan-blog-post)**.

I write about computer vision, applied machine learning, and the tools I use day to day —
structured as multi-part series rather than one-off posts.

## Series

- **Computer Vision Foundations** — from "what is an image" to first CV project
- **Anomaly Detection** — practical anomaly detection techniques
- **VLM Series** — working with vision-language models (e.g. Qwen3-14B)
- **Data Science Steps** — a working process for EDA through modeling
- **Command Line Mastery for HPC** — shell, SSH, and job scheduling for research computing
- **Vim Mastery** — editing efficiently from survival mode to advanced usage

## Local development

```bash
# preview with live reload
quarto preview

# render the full site
quarto render

# publish to quarto.pub
quarto publish quarto-pub
```

Some posts execute notebooks/code during render; see `requirements.txt` for the Python
dependencies used by the Computer Vision Foundations series.

## Structure

```
posts/series/   curated, multi-part series (the current, actively maintained content)
posts/          standalone posts
notebooks/      source notebooks for some series posts
_archive/       pre-Quarto legacy posts, kept for reference only (not built into the site)
```

## Links

- [About](https://hasangoni.quarto.pub/hasan-blog-post/about.html)
- [LinkedIn](https://www.linkedin.com/in/mohammed-hasan-goni-77614a89/)
- [GitHub](https://github.com/HasanGoni)
