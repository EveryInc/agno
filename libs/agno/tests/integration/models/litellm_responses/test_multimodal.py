import pytest
from pathlib import Path

from agno.agent import Agent, RunResponse
from agno.media import Audio, File, Image, Video
from agno.models.litellm import LiteLLMResponses


def _assert_metrics(response: RunResponse):
    """Helper function to assert metrics are present and valid"""
    # Check that metrics dictionary exists
    assert response.metrics is not None

    # Check that we have some token counts
    assert "input_tokens" in response.metrics
    assert "output_tokens" in response.metrics
    assert "total_tokens" in response.metrics

    # Check that we have timing information
    assert "time" in response.metrics

    # Check that the total tokens is the sum of input and output tokens
    input_tokens = sum(response.metrics.get("input_tokens", []))
    output_tokens = sum(response.metrics.get("output_tokens", []))
    total_tokens = sum(response.metrics.get("total_tokens", []))

    # The total should be at least the sum of input and output
    # (Note: sometimes there might be small discrepancies in how these are calculated)
    assert total_tokens >= input_tokens + output_tokens - 5  # Allow small margin of error


class TestMultimodalGPT5:
    """Test multimodal functionality with GPT-5 model"""

    def test_gpt5_image_analysis_no_reasoning(self):
        """Test GPT-5 image analysis without reasoning"""
        # Create a test image (you may need to adjust the path)
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Describe what you see in this image",
            images=[test_image]
        )

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        
        # Check that image was included in user message
        user_message = response.messages[1]
        assert user_message.images is not None
        assert len(user_message.images) == 1
        
        _assert_metrics(response)

    def test_gpt5_image_analysis_with_reasoning(self):
        """Test GPT-5 image analysis with reasoning enabled"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Analyze this image step by step and describe the visual elements, colors, and composition",
            images=[test_image]
        )

        assert response.content is not None
        
        # Check that reasoning was applied
        assistant_message = response.messages[-1]
        assert hasattr(assistant_message, 'reasoning_content') or hasattr(assistant_message, 'thinking')
        
        _assert_metrics(response)

    def test_gpt5_multiple_images_reasoning(self):
        """Test GPT-5 with multiple images and reasoning"""
        images = [
            Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"),
            Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png")
        ]

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "high", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Compare and contrast these two images. What are the main differences?",
            images=images
        )

        assert response.content is not None
        
        # Check that both images were included
        user_message = response.messages[1]
        assert user_message.images is not None
        assert len(user_message.images) == 2
        
        _assert_metrics(response)

    def test_gpt5_file_analysis_no_reasoning(self):
        """Test GPT-5 file analysis without reasoning"""
        # Create a simple test file
        test_file_path = Path("tmp/test_data.txt")
        test_file_path.parent.mkdir(exist_ok=True)
        test_file_path.write_text("Name,Age,City\nAlice,25,New York\nBob,30,London\nCharlie,35,Tokyo")
        
        test_file = File(filepath=str(test_file_path))

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Analyze this CSV file and tell me what data it contains",
            files=[test_file]
        )

        assert response.content is not None
        assert "Alice" in response.content or "CSV" in response.content or "data" in response.content.lower()
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_gpt5_async_multimodal_reasoning(self):
        """Test GPT-5 async multimodal with reasoning"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response = await agent.arun(
            "What's the mood or atmosphere conveyed by this image?",
            images=[test_image]
        )

        assert response.content is not None
        _assert_metrics(response)


class TestMultimodalClaude:
    """Test multimodal functionality with Claude Opus 4.1 model"""

    def test_claude_image_analysis_no_thinking(self):
        """Test Claude image analysis without thinking"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Describe what you see in this image",
            images=[test_image]
        )

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        
        # Check that image was included in user message
        user_message = response.messages[1]
        assert user_message.images is not None
        assert len(user_message.images) == 1
        
        _assert_metrics(response)

    def test_claude_image_analysis_with_thinking(self):
        """Test Claude image analysis with thinking enabled"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,  # Required when thinking is enabled
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Analyze this image in detail. What technical aspects can you observe?",
            images=[test_image]
        )

        assert response.content is not None
        _assert_metrics(response)

    def test_claude_thinking_high_budget_detailed_analysis(self):
        """Test Claude with high thinking budget for detailed image analysis"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2048  # High budget for detailed analysis
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Provide a comprehensive analysis of this image including composition, lighting, colors, subject matter, and potential photography techniques used",
            images=[test_image]
        )

        assert response.content is not None
        assert len(response.content) > 200  # Should be a detailed response
        _assert_metrics(response)

    def test_claude_multiple_images_thinking(self):
        """Test Claude with multiple images and thinking"""
        images = [
            Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"),
            Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png")
        ]

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1536
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Compare these images and identify the key visual differences and similarities",
            images=images
        )

        assert response.content is not None
        
        # Check that both images were included
        user_message = response.messages[1]
        assert user_message.images is not None
        assert len(user_message.images) == 2
        
        _assert_metrics(response)

    def test_claude_file_analysis_thinking(self):
        """Test Claude file analysis with thinking"""
        # Create a test JSON file
        test_file_path = Path("tmp/test_config.json")
        test_file_path.parent.mkdir(exist_ok=True)
        test_file_path.write_text('{"database": {"host": "localhost", "port": 5432}, "api": {"version": "v1", "timeout": 30}}')
        
        test_file = File(filepath=str(test_file_path))

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Analyze this configuration file and explain what it configures and suggest any improvements",
            files=[test_file]
        )

        assert response.content is not None
        assert "database" in response.content.lower() or "configuration" in response.content.lower()
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_claude_async_multimodal_thinking(self):
        """Test Claude async multimodal with thinking"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response = await agent.arun(
            "What story might this image tell? Use your imagination.",
            images=[test_image]
        )

        assert response.content is not None
        _assert_metrics(response)


class TestMultimodalStreaming:
    """Test streaming multimodal functionality with both models"""

    def test_gpt5_stream_image_analysis_reasoning(self):
        """Test GPT-5 streaming image analysis with reasoning"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response_stream = agent.run(
            "Describe this image in detail, focusing on the natural elements",
            images=[test_image],
            stream=True
        )

        # Verify it's an iterator
        assert hasattr(response_stream, "__iter__")

        responses = list(response_stream)
        assert len(responses) > 0
        for response in responses:
            assert response.content is not None

        _assert_metrics(agent.run_response)

    def test_claude_stream_image_analysis_thinking(self):
        """Test Claude streaming image analysis with thinking"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response_stream = agent.run(
            "Analyze the technical aspects of this image",
            images=[test_image],
            stream=True
        )

        # Verify it's an iterator
        assert hasattr(response_stream, "__iter__")

        responses = list(response_stream)
        assert len(responses) > 0
        for response in responses:
            assert response.content is not None

        _assert_metrics(agent.run_response)


class TestComplexMultimodalScenarios:
    """Test complex multimodal scenarios with both models"""

    def test_gpt5_image_and_file_reasoning(self):
        """Test GPT-5 with both image and file inputs plus reasoning"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
        
        # Create a test description file
        test_file_path = Path("tmp/image_metadata.txt")
        test_file_path.parent.mkdir(exist_ok=True)
        test_file_path.write_text("Location: Wisconsin, Madison\nPhoto taken: Summer 2019\nWeather: Clear, sunny day\nCamera: Canon EOS 5D")
        
        test_file = File(filepath=str(test_file_path))

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "high", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Compare the image with the metadata file. Do they match? Explain your reasoning.",
            images=[test_image],
            files=[test_file]
        )

        assert response.content is not None
        assert "wisconsin" in response.content.lower() or "madison" in response.content.lower()
        _assert_metrics(response)

    def test_claude_thinking_multimodal_analysis(self):
        """Test Claude with thinking for comprehensive multimodal analysis"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
        
        # Create analysis requirements file
        test_file_path = Path("tmp/analysis_requirements.txt")
        test_file_path.parent.mkdir(exist_ok=True)
        test_file_path.write_text("Analysis Requirements:\n1. Identify main subject\n2. Describe lighting conditions\n3. Evaluate composition\n4. Suggest improvements\n5. Rate overall quality 1-10")
        
        test_file = File(filepath=str(test_file_path))

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2048  # High budget for comprehensive analysis
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run(
            "Follow the analysis requirements in the file to evaluate this image",
            images=[test_image],
            files=[test_file]
        )

        assert response.content is not None
        # Should address the requirements from the file
        content_lower = response.content.lower()
        assert any(keyword in content_lower for keyword in ["lighting", "composition", "subject", "quality"])
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_async_complex_multimodal_both_models(self):
        """Test async complex multimodal with both models"""
        test_image = Image(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        # Test GPT-5
        gpt5_agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        gpt5_response = await gpt5_agent.arun(
            "What can you tell me about the technical aspects of this image?",
            images=[test_image]
        )

        assert gpt5_response.content is not None
        _assert_metrics(gpt5_response)

        # Test Claude
        claude_agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        claude_response = await claude_agent.arun(
            "What can you tell me about the technical aspects of this image?",
            images=[test_image]
        )

        assert claude_response.content is not None
        _assert_metrics(claude_response)

        # Both should have analyzed the image
        assert len(gpt5_response.content) > 10
        assert len(claude_response.content) > 10