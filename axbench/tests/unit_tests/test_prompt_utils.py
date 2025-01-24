import unittest
from unittest.mock import MagicMock, patch
import asyncio
from axbench.utils.prompt_utils import get_concept_genres
from axbench.utils.constants import EMPTY_CONCEPT

class TestPromptUtils(unittest.TestCase):
    """
    Unit tests for the prompt_utils module
    
    p.s. I am not sure how useful they are, as LLM-based API calls are hard to test.
    I think looking at the raw generation is the best way to go.

    So, I stop prompting cursor to generate more tests after all these. Bye!
    """
    def setUp(self):
        # Create mock client
        self.mock_client = MagicMock()
        
        # Mock concepts to test
        self.test_concepts = ["math", "programming"]
        
        # Mock API responses for different concepts
        self.mock_responses = [
            "text, math",  # Response for "math"
            "text, code"   # Response for "programming"
        ]

    @patch('axbench.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    async def test_get_concept_genres(self):
        """Test get_concept_genres function"""
        # Configure mock client's chat_completions method
        self.mock_client.chat_completions = MagicMock()
        self.mock_client.chat_completions.return_value = self.mock_responses
        
        # Call the function
        result = await get_concept_genres(
            client=self.mock_client,
            concepts=self.test_concepts,
            api_tag="test"
        )
        
        # Verify the client was called correctly
        expected_prompts = [
            "mock_prompt_math",
            "mock_prompt_programming"
        ]
        self.mock_client.chat_completions.assert_called_once_with(
            "test.get_concept_genre",
            expected_prompts
        )
        
        # Verify the results
        self.assertEqual(result["math"], ["text", "math"])
        self.assertEqual(result["programming"], ["text", "code"])

    @patch('axbench.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    async def test_get_concept_genres_none_response(self):
        """Test get_concept_genres function when LLM responds with 'none'"""
        # Configure mock client with 'none' response
        self.mock_client.chat_completions = MagicMock()
        self.mock_client.chat_completions.return_value = ["none", "none"]
        
        # Call the function
        result = await get_concept_genres(
            client=self.mock_client,
            concepts=self.test_concepts,
            api_tag="test"
        )
        
        # Verify results default to ["text"] when response is "none"
        self.assertEqual(result["math"], ["text"])
        self.assertEqual(result["programming"], ["text"])

    @patch('axbench.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    async def test_get_concept_genres_empty_concepts(self):
        """Test get_concept_genres function with empty concepts list"""
        result = await get_concept_genres(
            client=self.mock_client,
            concepts=[],
            api_tag="test"
        )
        
        # Verify empty dict is returned for empty concepts list
        self.assertEqual(result, {})
        # Verify client wasn't called
        self.mock_client.chat_completions.assert_not_called()

def run_async_test(coro):
    return asyncio.run(coro)

if __name__ == '__main__':
    # Helper to run async tests
    original_setUp = TestPromptUtils.setUp
    def async_setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        original_setUp(self)
    TestPromptUtils.setUp = async_setUp

    original_tearDown = TestPromptUtils.tearDown
    def async_tearDown(self):
        self.loop.close()
        if original_tearDown:
            original_tearDown(self)
    TestPromptUtils.tearDown = async_tearDown

    unittest.main()