from pathlib import Path
import hashlib
import unittest

from transformers import AutoTokenizer

from dataset.language_examples import TOKENIZER_PASSAGES, TRANSLATION_PAIRS
from scripts.eval_toolcall import TEST_CASES
from trainer.train_agent import execute_tool

ROOT = Path(__file__).resolve().parents[1]


class EnglishEditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)

    def test_released_tokenizer_and_chat_template(self):
        fingerprints = {
            "tokenizer.json": "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668",
            "tokenizer_config.json": "d7cd6a60c9f191c4f9ffec69b2eb289ad2f960366724f328b681ae1e46e4c110",
        }
        for filename, expected in fingerprints.items():
            with self.subTest(filename=filename):
                content = (ROOT / "model" / filename).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

        messages = [{"role": "user", "content": "Hello!"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("<|im_start|>user\nHello!<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_tokenizer_examples(self):
        tokens = self.tokenizer.encode("The sun sets in the west", add_special_tokens=False)
        self.assertEqual(tokens, [750, 3150, 2139, 118, 353, 309, 339, 611])
        for source, english in TOKENIZER_PASSAGES:
            self.assertNotEqual(source, english)
            for text in (source, english):
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_translation_tools(self):
        source, _ = TRANSLATION_PAIRS[0]
        self.assertTrue(any(source in case["prompt"] for case in TEST_CASES))
        for chinese, english in TRANSLATION_PAIRS:
            with self.subTest(english=english):
                self.assertNotEqual(chinese, english)
                to_english = execute_tool("translate_text", {"text": chinese, "target_language": "english"})
                to_chinese = execute_tool("translate_text", {"text": english, "target_language": "chinese"})
                self.assertEqual(to_english["translated_text"], english)
                self.assertEqual(to_chinese["translated_text"], chinese)

    def test_weather_for_english_inputs(self):
        expected = {"Beijing": ("28°C", "sunny"), "London": ("5°C", "light rain")}
        for city, (temperature, condition) in expected.items():
            with self.subTest(city=city):
                result = execute_tool("get_current_weather", {"location": city})
                self.assertEqual(result["temperature"], temperature)
                self.assertEqual(result["condition"], condition)


if __name__ == "__main__":
    unittest.main()
