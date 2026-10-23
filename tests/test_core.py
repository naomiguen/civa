"""
Uji unit untuk logika murni CIVA (tanpa model ML, jaringan, maupun GROBID asli).

Jalankan dari folder proyek:   python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import retrieval, verdict  # noqa: E402
from backend.grobid_client import parse_tei  # noqa: E402
from backend.schemas import Evidence, ParsedReference, Record  # noqa: E402
from backend.textutil import (author_overlap, contains_title, extract_arxiv_id, is_identifier_only,  # noqa: E402
                              norm_key, normalize_doi, paren_year, prepare_for_grobid, restore_title_case,
                              split_references, title_from_raw, title_similarity, title_word_issue,
                              titlecase_caps_runs)
from civa_core import authors_from_parts, clean_authors, clean_title, split_authors, to_initials  # noqa: E402


class TextUtilTests(unittest.TestCase):
    def test_norm_key(self):
        self.assertEqual(norm_key("Déjà Vu: A Study!"), "deja vu a study")

    def test_title_from_raw_when_grobid_finds_no_title(self):
        self.assertEqual(title_from_raw("Implementasi Dbscan Untuk Pengendalian Karhutla Di Provinsi Riau. (2025). "
                                        "Jurnal Ilmiah.", has_authors=False),
                         "Implementasi Dbscan Untuk Pengendalian Karhutla Di Provinsi Riau")  # APA tanpa penulis
        self.assertEqual(title_from_raw("Kellett, S. E. (2015). and War Memorial love: Australian the of windows "
                                        "Truth the. Jurnal Ilmiah. https://doi.org/10.1080/14443058.2015.1021366",
                                        has_authors=True),
                         "and War Memorial love: Australian the of windows Truth the")
        self.assertEqual(title_from_raw('Arifin, Z. (2020) “Sistem Pakar Diagnosa Hama,” Jurnal TI.', True),
                         "Sistem Pakar Diagnosa Hama")
        self.assertEqual(title_from_raw("tanpa tahun sama sekali", False), "")

    def test_title_word_issue_variants(self):
        record = "Moderate V4+/V5+ Redox Sites Modification Balances Efficient n-Butane Oxidation"
        self.assertEqual(title_word_issue("Efficient Oxidation Balances n-Butane Redox V4+/V5+ Moderate Sites "
                                          "Modification", record), "urutan judul")
        self.assertIsNone(title_word_issue("Assessing Cyber Security", "Assessing Cyber Security 2015"))
        self.assertTrue(contains_title("Rademaker, M. (2016). Assessing Cyber Security 2015. Jurnal.",
                                       "Assessing Cyber Security 2015"))
        self.assertFalse(contains_title("Logic. Jurnal.", "Logic"))  # judul terlalu pendek untuk dipercaya

    def test_title_similarity(self):
        self.assertEqual(title_similarity("Deep Learning, a Review", "deep learning a review"), 1.0)
        subtitle_dropped = title_similarity("Deep Learning for Natural Language Processing",
                                            "Deep Learning for Natural Language Processing: A Review")
        self.assertGreaterEqual(subtitle_dropped, 0.9)
        self.assertLess(title_similarity("Deep Learning for NLP", "Quantum Gravity in Loop Space"), 0.6)

    def test_normalize_doi_survives_copy_paste_artifacts(self):
        plain = "10.30598/barekengvol19iss4pp2765-2776"
        for dash in ("‑", "–", "−", "‐"):  # non-breaking hyphen, en dash, minus, hyphen
            self.assertEqual(normalize_doi(f"https://doi.org/10.30598/barekengvol19iss4pp2765{dash}2776."), plain)
        self.assertEqual(normalize_doi("doi: 10.30598/barekeng­vol19iss4pp2765-2776​"), plain)
        self.assertEqual(normalize_doi("https://doi.org/10.1000/abc”."), "10.1000/abc")

    def test_paren_year_and_grobid_preparation(self):
        self.assertEqual(paren_year("Utari, D.T. dkk. (2025) “Judul,” 19(4), hal. 27-76."), 2025)
        self.assertEqual(paren_year("Smith (2020a). Title."), 2020)
        self.assertIsNone(paren_year("Judul periode 1995 2024"))
        prepared = prepare_for_grobid("Ulhaq, N.D. dan Haryanto, Y.D. (2022) “Judul dan Isi,” hal. 69-77. "
                                      "Tersedia pada: https://doi.org/10.1/x")
        self.assertIn("Ulhaq, N.D. & Haryanto, Y.D. (2022)", prepared)
        self.assertIn("Judul dan Isi", prepared)  # 'dan' di dalam judul TIDAK diubah
        self.assertIn("pp. 69-77. Available at: https://", prepared)
        self.assertIn("Utari, D.T. et al. (2025)", prepare_for_grobid("Utari, D.T. dkk. (2025) Judul."))
        self.assertEqual(prepare_for_grobid("Tanpa tahun dan dkk."), "Tanpa tahun dan dkk.")

    def test_all_caps_title_is_titlecased_for_grobid_and_restored_after(self):
        raw = "Efendi, A. (2022). PENGEMBANGAN MULTIMEDIA TEKNIK DASAR RENANG DI KABUPATEN KUDUS. Jurnal Penjaskesrek."
        prepared = prepare_for_grobid(raw)
        self.assertIn("Pengembangan Multimedia Teknik Dasar Renang Di Kabupaten Kudus", prepared)
        self.assertIn("Efendi, A.", prepared)  # nama penulis & bagian lain tidak disentuh
        restored = restore_title_case("Pengembangan Multimedia Teknik Dasar Renang Di Kabupaten Kudus", raw)
        self.assertEqual(restored, "PENGEMBANGAN MULTIMEDIA TEKNIK DASAR RENANG DI KABUPATEN KUDUS")
        # akronim pendek / kapital tunggal tidak diubah
        self.assertEqual(titlecase_caps_runs("Metode CNN dan LSTM pada IEEE"), "Metode CNN dan LSTM pada IEEE")
        self.assertEqual(restore_title_case("Tidak Ada Di Teks", "teks lain"), "Tidak Ada Di Teks")

    def test_identifier_only_detection_and_arxiv_ids(self):
        self.assertTrue(is_identifier_only("https://doi.org/10.30598/barekengvol19iss4pp2765-2776"))
        self.assertTrue(is_identifier_only("doi: 10.1109/CVPR.2016.90"))
        self.assertTrue(is_identifier_only("https://arxiv.org/abs/1706.03762v7"))
        self.assertTrue(is_identifier_only("arXiv:1706.03762"))
        self.assertFalse(is_identifier_only(
            "Utari, D.T. (2025) Comparative Analysis of Machine Learning Models. https://doi.org/10.30598/x-1"))
        self.assertFalse(is_identifier_only("Judul tanpa pengenal sama sekali"))
        self.assertEqual(extract_arxiv_id("https://arxiv.org/pdf/1706.03762v7.pdf"), "1706.03762")
        self.assertEqual(extract_arxiv_id("10.48550/arXiv.1706.03762"), "1706.03762")
        self.assertEqual(extract_arxiv_id("arxiv.org/abs/cs/0112017"), "cs/0112017")
        self.assertIsNone(extract_arxiv_id("https://doi.org/10.1000/182"))

    def test_normalize_doi(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC.1)"), "10.1000/abc.1")
        self.assertEqual(normalize_doi("doi:10.5555/xyz,"), "10.5555/xyz")
        self.assertIsNone(normalize_doi("9783658384425"))
        self.assertIsNone(normalize_doi(None))

    def test_author_overlap_is_initial_aware(self):
        ref = [("J", "Smith"), ("R", "Patel")]
        self.assertEqual(author_overlap(ref, ["John Smith", "Raj Patel"]), 1.0)
        self.assertEqual(author_overlap(ref, ["Alice Smith", "Raj Patel"]), 0.5)  # "Smith, J." != "Alice Smith"
        self.assertEqual(author_overlap(ref, ["Wei Zhang", "Li Chen"]), 0.0)

    def test_author_overlap_name_order_and_particles(self):
        self.assertEqual(author_overlap([("Y", "Zhang")], ["Zhang Yue"]), 1.0)  # urutan nama terbalik
        self.assertEqual(author_overlap([("", "van der Berg")], ["Jan Berg"]), 1.0)  # nama belakang multi-kata
        self.assertEqual(author_overlap([("Y", "Zhang")], ["Wei Zhang 0001"]), 0.0)  # sufiks DBLP diabaikan, inisial beda
        self.assertIsNone(author_overlap([], ["John Smith"]))
        self.assertIsNone(author_overlap([("J", "Smith")], []))

    def test_split_references_numbered_with_wrapped_lines(self):
        text = "1. Zhang, Y. & Chen, L. (2023). Deep Learning\nfor NLP: A Review.\n2. Smith, J. (2024). Neural Nets."
        self.assertEqual(split_references(text), [
            "Zhang, Y. & Chen, L. (2023). Deep Learning for NLP: A Review.", "Smith, J. (2024). Neural Nets."])

    def test_split_references_blank_line_separated(self):
        self.assertEqual(split_references("[1] Aaa, B. (2020). Title one\ncontinues.\n\n[2] Ccc, D. (2021). Title two."),
                         ["Aaa, B. (2020). Title one continues.", "Ccc, D. (2021). Title two."])

    def test_split_references_unnumbered_hard_wrapped(self):
        # Kasus nyata: Harvard berbahasa Indonesia, tanpa nomor, tiap referensi terpotong beberapa baris.
        text = (
            "Ulhaq, N.D. dan Haryanto, Y.D. (2022) “Pemanfaatan Data Satelit Cuaca \n"
            "Himawari-8 dan Radiosonde dalam Analisis Hujan Lebat (Studi Kasus: \n"
            "Cilacap, 13 Januari 2021),” Jurnal Penelitian Sains, 24(2), hal. 69–77. \n"
            "Tersedia pada: https://doi.org/10.56064/jps.v24i2.671. \n"
            "Utari, D.T. dkk. (2025) “Comparative Analysis of Machine Learning Models for \n"
            "Rainfall Classification in Yogyakarta,” 19(4), hal. 2765–2776. Tersedia pada: \n"
            "https://doi.org/10.30598/barekengvol19iss4pp2765-2776. \n"
            "Valennur, M.A. dkk. (2026) “Pengaruh ENSO terhadap Curah Hujan Tahunan di \n"
            "Wilayah Urban Balikpapan Berdasarkan Oceanic Niño Indeks Periode 1995\n"
            "2024,” Jurnal Laut Khatulistiwa, 9, hal. 29–35. Tersedia pada: \n"
            "https://doi.org/10.26418/lkuntan.v9i1.104008.")
        refs = split_references(text)
        self.assertEqual(len(refs), 3)
        self.assertTrue(refs[0].startswith("Ulhaq, N.D."))
        self.assertTrue(refs[0].endswith("10.56064/jps.v24i2.671."))
        self.assertTrue(refs[1].startswith("Utari, D.T."))
        self.assertTrue(refs[2].endswith("lkuntan.v9i1.104008."))

    def test_doi_broken_across_lines_is_rejoined_without_space(self):
        text = ("1. Utari, D.T. (2025) Judul Panjang. Tersedia pada: https://doi.org/10.30598/barekengvol19iss4pp2765-\n"
                "2776.\n2. Lain, X. (2024) Judul Lain.")
        refs = split_references(text)
        self.assertEqual(len(refs), 2)
        self.assertTrue(refs[0].endswith("pp2765-2776."))
        self.assertEqual(normalize_doi(refs[0]), "10.30598/barekengvol19iss4pp2765-2776")

    def test_author_list_wrapped_over_lines_is_not_split(self):
        text = ("Ulhaq, N.D., Haryanto, Y.D., Wijaya, A.,\nSetiawan, B. (2022) Judul Panjang. Jurnal A.\n"
                "Utari, D.T. (2025) Judul Lain. Jurnal B.")
        self.assertEqual(len(split_references(text)), 2)

    def test_split_references_one_per_line_and_heading(self):
        self.assertEqual(split_references("Daftar Pustaka\nRef satu 2020\nRef dua 2021"), ["Ref satu 2020", "Ref dua 2021"])
        self.assertEqual(split_references("   \n\n "), [])
        self.assertEqual(split_references("(1) Hanya satu referensi tahun 2019"), ["Hanya satu referensi tahun 2019"])


class GrobidTeiTests(unittest.TestCase):
    # Contoh langsung dari dokumentasi GROBID: <persName> membawa xmlns sendiri, <biblStruct> tidak.
    DOC_EXAMPLE = (
        '<biblStruct > <analytic> <title/> <author> <persName xmlns= "http://www.tei-c.org/ns/1.0" >'
        '<surname> Graff </surname></persName> </author> </analytic> <monogr> <title level= "j" > Expert. Opin. '
        'Ther. Targets </title> <imprint> <biblScope unit= "volume" > 6 </biblScope> <biblScope unit= "issue" > 1 '
        '</biblScope> <biblScope unit= "page" from= "103" to= "113" /> <date type= "published" when= "2002" /> '
        '</imprint> </monogr> </biblStruct>')

    ARTICLE = (
        '<biblStruct><analytic><title level="a" type="main">Deep Learning for NLP: A Review</title>'
        '<author><persName xmlns="http://www.tei-c.org/ns/1.0"><forename type="first">Y</forename>'
        '<surname>Zhang</surname></persName></author><author><persName xmlns="http://www.tei-c.org/ns/1.0">'
        '<forename type="first">Lei</forename><forename type="middle">K</forename><surname>Chen</surname></persName>'
        '</author></analytic><monogr><title level="j">Journal of AI</title><imprint>'
        '<date type="published" when="2023-05-01"/></imprint></monogr><idno type="DOI">10.1234/ABCD.5</idno>'
        '</biblStruct>')

    def test_documentation_example_has_no_article_title(self):
        parsed = parse_tei(self.DOC_EXAMPLE, "raw")
        self.assertEqual(parsed.title, "")  # hanya nama jurnal -> tidak ada judul untuk diverifikasi
        self.assertEqual(parsed.venue, "Expert. Opin. Ther. Targets")
        self.assertEqual(parsed.year, 2002)
        self.assertEqual(parsed.authors, [("", "Graff")])

    def test_article_with_namespaced_persname(self):
        parsed = parse_tei(self.ARTICLE.encode("utf-8"), "raw")
        self.assertEqual(parsed.title, "Deep Learning for NLP: A Review")
        self.assertEqual(parsed.authors, [("Y", "Zhang"), ("Lei K", "Chen")])
        self.assertEqual((parsed.year, parsed.doi, parsed.kind, parsed.venue), (2023, "10.1234/abcd.5", "journal", "Journal of AI"))

    def test_book_monograph(self):
        xml = ('<biblStruct><monogr><title level="m" type="main">Concrete Mathematics</title><author><persName>'
               '<forename>R</forename><surname>Graham</surname></persName></author><imprint>'
               '<date type="published" when="1989"/></imprint></monogr></biblStruct>')
        parsed = parse_tei(xml)
        self.assertEqual((parsed.title, parsed.kind, parsed.year), ("Concrete Mathematics", "book", 1989))
        self.assertEqual(parsed.authors, [("R", "Graham")])

    def test_full_tei_document_and_proceedings(self):
        xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><div><listBibl><biblStruct><analytic>'
               '<title level="a">A Paper</title></analytic><monogr><title level="m">Proceedings of the 5th Workshop '
               'on X</title></monogr></biblStruct></listBibl></div></back></text></TEI>')
        parsed = parse_tei(xml)
        self.assertEqual((parsed.title, parsed.kind), ("A Paper", "proceedings"))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_tei("not xml at all"))
        self.assertIsNone(parse_tei("<foo><bar/></foo>"))


def make_ref(**kw) -> ParsedReference:
    base = dict(raw="r", title="Deep Learning for Natural Language Processing: A Review",
                authors=[("Y", "Zhang"), ("L", "Chen")], year=2023)
    return ParsedReference(**{**base, **kw})


def make_record(**kw) -> Record:
    base = dict(title="Deep Learning for Natural Language Processing: A Review",
                authors=["Yue Zhang", "Lei Chen"], year=2023, source="crossref")
    return Record(**{**base, **kw})


class CompareTests(unittest.TestCase):
    def test_verified(self):
        ev = retrieval.compare(make_ref(), make_record(), "crossref", cached=False)
        self.assertEqual((ev.status, ev.author_match, ev.year_match), ("verified", True, True))

    def test_year_tolerance_preprint_vs_published(self):
        self.assertEqual(retrieval.compare(make_ref(year=2022), make_record(), "x", False).status, "verified")

    def test_mismatch_fields(self):
        ev = retrieval.compare(make_ref(year=2019), make_record(authors=["Wei Wang", "Kai Li"]), "x", False)
        self.assertEqual((ev.status, ev.mismatch_fields), ("mismatch", ["penulis", "tahun"]))

    def test_doi_mismatch(self):
        ev = retrieval.compare(make_ref(doi="10.1234/aaa"), make_record(doi="10.1234/bbb"), "x", False)
        self.assertEqual((ev.status, ev.mismatch_fields), ("mismatch", ["DOI"]))

    def test_similar_title_with_other_authors_is_not_an_accusation(self):
        # "Survey" vs "Review": makalah lain dengan judul mirip bukan bukti manipulasi.
        record = make_record(title="Deep Learning for Natural Language Processing: A Survey", authors=["Wei Wang"])
        ev = retrieval.compare(make_ref(), record, "x", False)
        self.assertEqual(ev.status, "not_found")
        self.assertGreaterEqual(ev.title_similarity, 0.90)

    def test_cs_hint_follows_record_flag(self):
        self.assertTrue(retrieval.compare(make_ref(), make_record(cs_flag=True), "x", False).cs_hint)
        self.assertFalse(retrieval.compare(make_ref(), make_record(), "x", False).cs_hint)

    def test_unrelated_title_not_found(self):
        self.assertEqual(retrieval.compare(make_ref(), make_record(title="Quantum Gravity in Loop Space"), "x", False)
                         .status, "not_found")

    def test_valid_doi_forgives_doi_difference(self):
        # Judul + penulis + tahun cocok, DOI berbeda, tetapi DOI pengguna terbukti sah (mis. versi preprint).
        ev = retrieval.compare(make_ref(doi="10.1234/aaa"), make_record(doi="10.1234/bbb"), "x", False)
        self.assertEqual((ev.status, ev.mismatch_fields), ("mismatch", ["DOI"]))
        forgiven = retrieval._forgive_doi(ev)
        self.assertEqual((forgiven.status, forgiven.mismatch_fields), ("verified", []))
        # DOI yang dimaafkan tidak menghapus ketidakcocokan lain
        ev2 = retrieval.compare(make_ref(doi="10.1234/aaa", year=2010), make_record(doi="10.1234/bbb"), "x", False)
        self.assertEqual(retrieval._forgive_doi(ev2).mismatch_fields, ["tahun"])

    def test_missing_metadata_is_neutral(self):
        ev = retrieval.compare(make_ref(year=None), make_record(authors=[]), "x", False)
        self.assertEqual((ev.status, ev.author_match, ev.year_match), ("verified", None, None))

    def test_affiliation_noise_in_record_authors_is_ignored(self):
        record = make_record(authors=["Yue Zhang", "Lei Chen", "Department of Computer Science, Universitas Indonesia"])
        self.assertEqual(retrieval.compare(make_ref(), record, "x", False).status, "verified")

    def test_manipulated_title_words(self):
        shuffled = make_ref(title="Processing Language Natural for Learning Deep: Review A")
        ev = retrieval.compare(shuffled, make_record(), "x", False)
        self.assertEqual((ev.status, ev.mismatch_fields), ("mismatch", ["urutan judul"]))
        dropped = make_ref(title="Deep Learning for Language Processing: A Review")
        self.assertEqual(retrieval.compare(dropped, make_record(), "x", False).mismatch_fields, ["kata judul"])

    def test_honest_title_variants_stay_verified(self):
        for title in ("Deep Learning for Natural Language Processing",  # subjudul tidak ditulis
                      "deep learning for natural language processing - a review"):
            self.assertEqual(retrieval.compare(make_ref(title=title), make_record(), "x", False).status, "verified")
        record = make_record(title="The Ethics & Governance of Machine Learning Systems in Healthcare")
        ref = make_ref(title="Ethics and Governance of Machine Learning Systems in Healthcare")
        self.assertEqual(retrieval.compare(ref, record, "x", False).status, "verified")  # kata sandang & '&'

    def test_manipulated_authors(self):
        cases = {
            "tanpa penulis": make_ref(authors=[]),
            "susunan nama": make_ref(authors=[("Z", "Yue"), ("C", "Lei")]),  # "Yue, Z." untuk Yue Zhang
            "jumlah penulis": make_ref(authors=[("Y", "Zhang")]),  # satu dari dua penulis, tanpa "et al."
        }
        for field, ref in cases.items():
            ev = retrieval.compare(ref, make_record(), "x", False)
            self.assertEqual((ev.status, ev.mismatch_fields, ev.author_match), ("mismatch", [field], False), field)

    def test_first_name_only_author(self):
        ev = retrieval.compare(make_ref(authors=[("Y", "Zhang"), ("", "Lei")]), make_record(), "x", False)
        self.assertEqual(ev.mismatch_fields, ["nama penulis"])  # "Lei" untuk Lei Chen
        # nama tunggal yang cocok dgn nama belakang (atau nama kembar "Saliman Saliman") wajar
        ok = retrieval.compare(make_ref(authors=[("Y", "Zhang"), ("", "Saliman")]),
                               make_record(authors=["Yue Zhang", "Saliman Saliman"]), "x", False)
        self.assertEqual(ok.status, "verified")

    def test_author_coverage_and_name_order(self):
        record = make_record(authors=["Reynold F. Nesiba", "David Sorenson", "Kaleb Sturm"])
        merged = make_ref(authors=[("", "Nesiba"), ("", "Sorenson"), ("D S K F", "Reynold")])  # Sturm hilang
        self.assertEqual(retrieval.compare(merged, record, "x", False).mismatch_fields, ["jumlah penulis"])
        flipped = make_ref(authors=[("R", "Widyastuti"), ("S R P", "Listia")])  # "Listia, S. R. P."
        record = make_record(authors=["Reni Widyastuti", "Listia Sari Puspita"])
        self.assertEqual(retrieval.compare(flipped, record, "x", False).mismatch_fields, ["susunan nama"])
        middle = make_ref(authors=[("", "Panita")])
        record = make_record(authors=["Singgih Panita Budiman"])
        self.assertEqual(retrieval.compare(middle, record, "x", False).mismatch_fields, ["nama penulis"])
        # nama belakang pertama orang Spanyol + inisial yang cocok tetap wajar
        spanish = make_ref(authors=[("J", "Pérez")])
        self.assertEqual(retrieval.compare(spanish, make_record(authors=["José Pérez García"]), "x", False).status,
                         "verified")

    def test_grobid_truncated_title_is_not_a_manipulation(self):
        record = make_record(title="Karakteristik Sohun Pati Aren dengan Penambahan Ekstrak Umbi Bit, Daun Suji, dan Kunyit")
        ref = make_ref(title="Karakteristik Sohun Pati Aren dengan Penambahan Ekstrak Umbi Bit",
                       raw="Zhang, Y., & Chen, L. (2023). KARAKTERISTIK SOHUN PATI AREN DENGAN PENAMBAHAN EKSTRAK "
                           "UMBI BIT, DAUN SUJI, DAN KUNYIT. Jurnal Pangan.")
        self.assertEqual(retrieval.compare(ref, record, "x", False).status, "verified")
        ref.raw = "r"  # judul memang ditulis terpotong oleh pengguna
        self.assertEqual(retrieval.compare(ref, record, "x", False).mismatch_fields, ["kata judul"])

    def test_shortened_author_list_rules(self):
        record = make_record(authors=["Wei Wang", "Xin Zhu", "Gang Zhou", "Kai Li", "Ming Sun", "Hao Chen"])
        first_four = make_ref(authors=[("W", "Wang"), ("X", "Zhu"), ("G", "Zhou"), ("K", "Li")])
        self.assertEqual(retrieval.compare(first_four, record, "x", False).status, "verified")  # dipersingkat wajar
        skipped = make_ref(authors=[("W", "Wang"), ("G", "Zhou"), ("K", "Li")])  # Xin Zhu dilompati
        self.assertEqual(retrieval.compare(skipped, record, "x", False).mismatch_fields, ["jumlah penulis"])
        only_first = make_ref(authors=[("W", "Wang")])  # tinggal satu nama
        self.assertEqual(retrieval.compare(only_first, record, "x", False).mismatch_fields, ["jumlah penulis"])

    def test_et_al_allows_shortened_author_list(self):
        ref = make_ref(authors=[("Y", "Zhang")], raw="Zhang, Y. et al. (2023). Deep Learning for NLP: A Review.")
        self.assertEqual(retrieval.compare(ref, make_record(), "x", False).status, "verified")

    def test_best_of_prefers_verified(self):
        good = retrieval.compare(make_ref(), make_record(), "a", False)
        bad = retrieval.compare(make_ref(year=1999), make_record(), "b", False)
        self.assertIs(retrieval.best_of([bad, good]), good)
        self.assertIsNone(retrieval.best_of([Evidence(status="not_found")]))


class VerdictTests(unittest.TestCase):
    def setUp(self):  # bobot tetap, tidak bergantung pada environment variable
        patcher = mock.patch.object(verdict, "settings", SimpleNamespace(evidence_weight=4.0, not_found_weight=-1.5))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_verified_lifts_probability(self):
        status, confidence, _ = verdict.fuse(0.5, Evidence(status="verified"))
        self.assertEqual(status, "valid")
        self.assertGreater(confidence, 0.97)

    def test_mismatch_overrides_moderately_confident_ml(self):
        status, confidence, _ = verdict.fuse(0.9, Evidence(status="mismatch"))
        self.assertEqual(status, "halu")
        self.assertGreater(confidence, 0.5)

    def test_not_found_leans_halu_unless_unreachable(self):
        self.assertEqual(verdict.fuse(0.6, Evidence(status="not_found"))[0], "halu")
        self.assertEqual(verdict.fuse(0.6, Evidence(status="not_found", unreachable=True))[0], "valid")

    def test_zero_weights_give_pure_ml(self):
        with mock.patch.object(verdict, "settings", SimpleNamespace(evidence_weight=0.0, not_found_weight=0.0)):
            _, confidence, final = verdict.fuse(0.73, Evidence(status="verified"))
        self.assertAlmostEqual(final, 0.73, places=6)
        self.assertAlmostEqual(confidence, 0.73, places=6)

    def test_reasons_match_design_text(self):
        ev = Evidence(status="verified", author_match=True, year_match=True, doi_match=True)
        self.assertEqual(verdict.build_reason("valid", ev),
                         "Alasan: Referensi terverifikasi; judul, penulis, tahun, dan DOI sesuai pangkalan data.")
        ev.doi_match = None
        self.assertEqual(verdict.build_reason("valid", ev),
                         "Alasan: Referensi terverifikasi; judul, penulis, dan tahun sesuai pangkalan data.")
        self.assertEqual(verdict.build_reason("halu", Evidence(status="mismatch", mismatch_fields=["tahun", "penulis"])),
                         "Alasan: Indikasi manipulasi AI. Judul ditemukan, namun tahun dan penulis tidak sesuai.")
        self.assertEqual(verdict.build_reason("out_of_scope", Evidence()),
                         "Referensi berada di luar batasan domain Ilmu Komputer (Out-of-Distribution).")

    def test_manipulation_reasons(self):
        ev = Evidence(status="mismatch", mismatch_fields=["urutan judul"])
        self.assertEqual(verdict.build_reason("halu", ev),
                         "Alasan: Indikasi manipulasi AI. Judul ditemukan, namun urutan kata judulnya diacak.")
        ev.mismatch_fields = ["kata judul", "jumlah penulis", "tahun"]
        self.assertEqual(verdict.build_reason("halu", ev),
                         "Alasan: Indikasi manipulasi AI. Judul ditemukan, namun ada kata judul yang dihilangkan, "
                         "sebagian penulis tidak dicantumkan, dan tahun tidak sesuai.")

    def test_doi_issue_reasons(self):
        missing = Evidence(status="mismatch", mismatch_fields=["DOI"], doi_issue="missing")
        self.assertIn("DOI yang dicantumkan tidak terdaftar", verdict.build_reason("halu", missing))
        other = Evidence(status="mismatch", mismatch_fields=["DOI"], doi_issue="other_paper",
                         record=make_record(title="Deep Residual Learning"))
        self.assertIn('mengarah ke makalah lain: "Deep Residual Learning"', verdict.build_reason("halu", other))

    def test_authors_display_apa_style(self):
        self.assertEqual(verdict.authors_display([("Y", "Zhang"), ("L", "Chen")]), "Zhang, Y. & Chen, L.")
        self.assertEqual(verdict.authors_display([("Lei K", "Chen")]), "Chen, L. K.")
        self.assertEqual(verdict.authors_display([("", "Graff")]), "Graff")
        self.assertEqual(verdict.authors_display([]), "Penulis tidak terdeteksi")
        self.assertTrue(verdict.authors_display([("A", "A1"), ("B", "B1"), ("C", "C1"), ("D", "D1")]).endswith("dkk."))

    def test_overall_percentages(self):
        # 6 valid + 1 di luar scope yang valid ; 2 halu + 1 di luar scope yang tidak valid
        self.assertEqual(verdict.overall_percentages(6 + 1, 2 + 1), (70.0, 30.0))
        self.assertEqual(verdict.overall_percentages(1, 2), (33.3, 66.7))
        self.assertEqual(verdict.overall_percentages(0, 0), (0.0, 0.0))  # semua tidak terbaca

    def test_join_id(self):
        self.assertEqual(verdict.join_id(["judul", "penulis", "tahun"]), "judul, penulis, dan tahun")
        self.assertEqual(verdict.join_id(["tahun", "penulis"]), "tahun dan penulis")


class CoreTextTests(unittest.TestCase):
    def test_clean_title(self):
        self.assertEqual(clean_title('  "DEEP   Learning for NLP: A Review." '), "deep learning for nlp: a review")
        self.assertEqual(clean_title("Judul ?", lowercase=False), "Judul ?")
        self.assertEqual(clean_title(None), "")

    def test_clean_authors_canonical_initials_and_affiliations(self):
        self.assertEqual(clean_authors("R.G. Krisnadhi"), "r. g. krisnadhi")
        self.assertEqual(clean_authors("Y Zhang, L. Chen & X Wang", lowercase=False), "Y. Zhang, L. Chen, X. Wang")
        self.assertEqual(clean_authors("Dian Pramadhana, Program Studi Teknik Informatika, Politeknik Baja Tegal",
                                       lowercase=False), "Dian Pramadhana")
        self.assertEqual(split_authors("Eko Yulianto3, Ricki Sastra4"), ["Eko Yulianto", "Ricki Sastra"])
        self.assertEqual(split_authors("George Dan Mogoşanu, Dan Jurafsky"), ["George Dan Mogoşanu", "Dan Jurafsky"])
        self.assertEqual(split_authors("Budi Santoso dan Ani Lestari"), ["Budi Santoso", "Ani Lestari"])

    def test_same_text_for_grobid_and_dataset_formats(self):
        # GROBID memberi 'Y' tanpa titik; dataset memakai 'Y.' -> setelah dibersihkan harus identik.
        self.assertEqual(clean_authors(authors_from_parts([("Y", "Zhang")])), clean_authors("Y. Zhang"))

    def test_to_initials(self):
        self.assertEqual(to_initials("Mehdi Sari, Joao Arwidarasto"), "M. Sari, J. Arwidarasto")
        self.assertEqual(to_initials("Madonna"), "Madonna")


class DatasetLinkTests(unittest.TestCase):
    """Penautan referensi halu ke sumber aslinya + jenis manipulasinya (train_model.py)."""

    SOURCES = [("Analisis Sentimen Berbasis Aspek pada Ulasan Hotel", "Yamka Sudirman, Dwi Simatupang"),
               ("Complete Multilingual Neural Machine Translation", "Markus Freitag, Orhan Firat"),
               ("Dispute Settlement in ASEAN", "Walter Woon")]

    def setUp(self):
        import train_model  # impor di sini: modul training memuat sklearn/xgboost
        self.tm = train_model

    def link(self, title, author):
        titles, authors = zip(*self.SOURCES)
        return int(self.tm.link_sources(titles, authors, [title], [author])[0])

    def test_link_sources(self):
        self.assertEqual(self.link("Dispute Settlement in ASEAN", "Woon Walter"), 2)  # judul sama
        self.assertEqual(self.link("Hotel pada Analisis Ulasan Aspek Sentimen Berbasis", "Yamka Sudirman"), 0)  # diacak
        self.assertEqual(self.link("Analisis Sentimen Aspek pada Hotel", "Yamka Sudirman, Dwi Simatupang"), 0)  # dihapus
        self.assertEqual(self.link("Multilingual Neural", "Markus Freitag, Orhan Firat"), 1)  # sisa 2 kata: lewat penulis
        self.assertEqual(self.link("Quantum Gravity in Loop Space", "Carlo Rovelli"), -1)

    def test_perturbation_type(self):
        title, author = self.SOURCES[0]
        kind = self.tm.perturbation_type
        self.assertEqual(kind(title, author, "Hotel pada Analisis Ulasan Aspek Sentimen Berbasis", author), "judul_diacak")
        self.assertEqual(kind(title, author, "Analisis Sentimen Aspek pada Hotel", author), "judul_kata_dihapus")
        self.assertEqual(kind(title, author, title, "Sudirman, Yamka Simatupang Dwi"), "nama_dibalik")
        self.assertEqual(kind(title, author, title, "Yamka Dwi"), "penulis_tidak_lengkap")
        self.assertEqual(kind(title, author, title, ""), "tanpa_penulis")
        self.assertEqual(kind(title, author, title.upper(), author), "identik")


if __name__ == "__main__":
    unittest.main()
