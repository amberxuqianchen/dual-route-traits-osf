"""
Copy preprocessed data files from sibling source repos into 1_data/.

Usage:
    python 2_code/copy_data.py
    python 2_code/copy_data.py --us-src /path/to/face-bert --cn-src /path/to/chinese-celeb

The Chinese first-substantive semantic text and embeddings are copied when the
source tree already contains them. This script copies existing acquisition
artifacts; it does not regenerate text embeddings.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

US_FILES = [
    # (relative path inside --us-src, destination inside 1_data/us/)
    ('face-bert-clean/pipeline/preprocessed/full_sample/ratings_clean.npy',
     'ratings_clean.npy'),
    ('face-bert-clean/pipeline/wikiRDM_GPT3large_18May2026/rdm_cosine.npy',
     'rdm_semantic_gpt_wiki_cosine.npy'),
    ('face-bert-clean/pipeline/vggface2_RDM/rdm_cosine.npy',
     'rdm_visual_vggface2_cosine.npy'),
]

US_TSNE_FILES = [
    ('wikiRDM_GPT3large_18May2026',
     'semantic_gpt_wiki_18May2026/embeddings'),
]

CN_REQUIRED_FILES = [
    ('3_pipeline/descriptive_behavioral/ratings_clean.csv',
     'ratings_clean.csv'),
    ('3_pipeline/RDM/RDM_Visual_VGGFace2_cosine.npy',
     'rdm_visual_vggface2_cosine.npy'),
]

CN_FIRST_SUBSTANTIVE_FILES = [
    ((
        '3_pipeline/RDM/RDM_Semantic_FirstSubstantive_large_cosine.npy',
        '3_pipeline/RDM/rdm_semantic_first_substantive_large_cosine.npy',
        '1_data/cn/rdm_semantic_first_substantive_large_cosine.npy',
    ), 'rdm_semantic_first_substantive_large_cosine.npy'),
    ((
        '3_pipeline/cn_first_substantive_text',
        '1_data/cn/semantic_first_substantive/text',
    ), 'semantic_first_substantive/text'),
    ((
        '3_pipeline/cn_first_substantive_embeddings',
        '1_data/cn/semantic_first_substantive/embeddings',
    ), 'semantic_first_substantive/embeddings'),
]

def _candidate_paths(src_root: Path, rel_sources):
    if isinstance(rel_sources, (str, Path)):
        rel_sources = (rel_sources,)
    return [src_root / rel_src for rel_src in rel_sources]


def _copy_path(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_entries(src_root: Path, entries: list, dest_dir: Path, required: bool = True):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel_sources, dest_name in entries:
        candidates = _candidate_paths(src_root, rel_sources)
        src = next((candidate for candidate in candidates if candidate.exists()), None)
        dst = dest_dir / dest_name

        if src is None:
            tried = ', '.join(str(candidate) for candidate in candidates)
            if required:
                raise FileNotFoundError(f"Source not found; tried: {tried}")
            print(f"  SKIP {dest_name} (not found; tried: {tried})")
            continue

        if src.resolve() == dst.resolve():
            print(f"  {dst} already in place")
            continue

        _copy_path(src, dst)
        print(f"  {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--us-src', default=str(ROOT.parent / 'face-bert'),
                        help='Path to face-bert repo root')
    parser.add_argument('--cn-src', default=str(ROOT.parent / 'chinese-celeb'),
                        help='Path to chinese-celeb repo root')
    args = parser.parse_args()

    us_src = Path(args.us_src)
    cn_src = Path(args.cn_src)

    print(f"Copying US data from {us_src}")
    copy_entries(us_src, US_FILES, ROOT / '1_data' / 'us')

    print(f"Copying US t-SNE semantic embeddings from {us_src}, when present")
    copy_entries(us_src, US_TSNE_FILES, ROOT / '1_data' / 'us', required=False)

    print("Extracting US vModule from raw .mat file")
    mat_path = us_src / 'data' / 'CelebARate_NT_online_orgnized.mat'
    dest_vmodule = ROOT / '1_data' / 'us' / 'vmodule.npy'
    if mat_path.exists():
        import numpy as np
        import scipy.io
        mat = scipy.io.loadmat(str(mat_path))
        np.save(dest_vmodule, mat['vModule'].ravel().astype(np.uint8))
        print(f"  {mat_path} -> {dest_vmodule}")
    else:
        print(f"  SKIP vmodule.npy (mat not found at {mat_path})")

    print(f"Copying CN required data from {cn_src}")
    copy_entries(cn_src, CN_REQUIRED_FILES, ROOT / '1_data' / 'cn')

    print(f"Copying CN first-substantive semantic artifacts from {cn_src}, when present")
    copy_entries(cn_src, CN_FIRST_SUBSTANTIVE_FILES, ROOT / '1_data' / 'cn', required=False)

    print("Done.")


if __name__ == '__main__':
    main()
