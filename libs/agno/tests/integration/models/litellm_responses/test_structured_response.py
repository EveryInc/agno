import pytest
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from agno.agent import Agent, RunResponse
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


class TestStructuredResponseGPT5:
    """Test structured response functionality with GPT-5 model"""

    def test_gpt5_simple_structured_response_no_reasoning(self):
        """Test GPT-5 simple structured response without reasoning"""
        class MovieSummary(BaseModel):
            title: str = Field(..., description="Movie title")
            genre: str = Field(..., description="Movie genre")
            year: int = Field(..., description="Release year")
            rating: float = Field(..., description="Rating out of 10")

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
            response_model=MovieSummary,
        )

        response = agent.run("Create a summary for the movie 'The Matrix'")

        # Verify structured output
        assert isinstance(response.content, MovieSummary)
        assert response.content.title is not None
        assert response.content.genre is not None
        assert response.content.year > 1900
        assert 0 <= response.content.rating <= 10
        _assert_metrics(response)

    def test_gpt5_complex_structured_response_reasoning(self):
        """Test GPT-5 complex structured response with reasoning"""
        class BusinessPlan(BaseModel):
            business_name: str = Field(..., description="Name of the business")
            industry: str = Field(..., description="Industry sector")
            target_market: str = Field(..., description="Target market description")
            revenue_streams: List[str] = Field(..., description="List of revenue streams")
            startup_costs: float = Field(..., description="Estimated startup costs")
            monthly_expenses: float = Field(..., description="Estimated monthly expenses")
            break_even_months: int = Field(..., description="Estimated months to break even")

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
            response_model=BusinessPlan,
        )

        response = agent.run("Create a detailed business plan for a coffee shop in downtown area")

        # Verify complex structured output
        assert isinstance(response.content, BusinessPlan)
        assert response.content.business_name is not None
        assert response.content.industry is not None
        assert len(response.content.revenue_streams) > 0
        assert response.content.startup_costs > 0
        assert response.content.monthly_expenses > 0
        assert response.content.break_even_months > 0
        _assert_metrics(response)

    def test_gpt5_nested_structured_response_reasoning(self):
        """Test GPT-5 nested structured response with reasoning"""
        class Address(BaseModel):
            street: str
            city: str
            state: str
            zip_code: str

        class Contact(BaseModel):
            email: str
            phone: str

        class Employee(BaseModel):
            name: str
            position: str
            salary: float
            contact: Contact
            address: Address

        class Company(BaseModel):
            company_name: str = Field(..., description="Company name")
            employees: List[Employee] = Field(..., description="List of employees")
            headquarters: Address = Field(..., description="Company headquarters address")

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
            response_model=Company,
        )

        response = agent.run("Create a company structure with 3 employees for a tech startup")

        # Verify nested structured output
        assert isinstance(response.content, Company)
        assert response.content.company_name is not None
        assert len(response.content.employees) == 3
        
        for employee in response.content.employees:
            assert isinstance(employee, Employee)
            assert employee.name is not None
            assert employee.position is not None
            assert employee.salary > 0
            assert isinstance(employee.contact, Contact)
            assert isinstance(employee.address, Address)
            
        assert isinstance(response.content.headquarters, Address)
        _assert_metrics(response)

    def test_gpt5_optional_fields_reasoning(self):
        """Test GPT-5 structured response with optional fields and reasoning"""
        class Product(BaseModel):
            name: str = Field(..., description="Product name")
            price: float = Field(..., description="Product price")
            description: str = Field(..., description="Product description")
            category: str = Field(..., description="Product category")
            tags: Optional[List[str]] = Field(None, description="Optional product tags")
            discount: Optional[float] = Field(None, description="Optional discount percentage")
            availability: bool = Field(default=True, description="Product availability")

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
            response_model=Product,
        )

        response = agent.run("Create a product listing for a wireless bluetooth headphone")

        # Verify structured output with optional fields
        assert isinstance(response.content, Product)
        assert response.content.name is not None
        assert response.content.price > 0
        assert response.content.description is not None
        assert response.content.category is not None
        assert isinstance(response.content.availability, bool)
        # Optional fields can be None or have values
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_gpt5_async_structured_response(self):
        """Test GPT-5 async structured response"""
        class Recipe(BaseModel):
            name: str = Field(..., description="Recipe name")
            ingredients: List[str] = Field(..., description="List of ingredients")
            instructions: List[str] = Field(..., description="Step by step instructions")
            prep_time: int = Field(..., description="Preparation time in minutes")
            cook_time: int = Field(..., description="Cooking time in minutes")
            servings: int = Field(..., description="Number of servings")

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
            response_model=Recipe,
        )

        response = await agent.arun("Create a recipe for chocolate chip cookies")

        # Verify structured output
        assert isinstance(response.content, Recipe)
        assert response.content.name is not None
        assert len(response.content.ingredients) > 0
        assert len(response.content.instructions) > 0
        assert response.content.prep_time > 0
        assert response.content.cook_time > 0
        assert response.content.servings > 0
        _assert_metrics(response)


class TestStructuredResponseClaude:
    """Test structured response functionality with Claude Opus 4.1 model"""

    def test_claude_simple_structured_response_no_thinking(self):
        """Test Claude simple structured response without thinking"""
        class BookSummary(BaseModel):
            title: str = Field(..., description="Book title")
            author: str = Field(..., description="Book author")
            genre: str = Field(..., description="Book genre")
            pages: int = Field(..., description="Number of pages")
            summary: str = Field(..., description="Brief summary")

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
            response_model=BookSummary,
        )

        response = agent.run("Create a summary for the book '1984' by George Orwell")

        # Verify structured output
        assert isinstance(response.content, BookSummary)
        assert response.content.title is not None
        assert response.content.author is not None
        assert response.content.genre is not None
        assert response.content.pages > 0
        assert response.content.summary is not None
        _assert_metrics(response)

    def test_claude_complex_structured_response_thinking(self):
        """Test Claude complex structured response with thinking"""
        class MarketAnalysis(BaseModel):
            market_name: str = Field(..., description="Name of the market")
            market_size: float = Field(..., description="Market size in billions USD")
            growth_rate: float = Field(..., description="Annual growth rate percentage")
            key_players: List[str] = Field(..., description="List of key market players")
            opportunities: List[str] = Field(..., description="Market opportunities")
            threats: List[str] = Field(..., description="Market threats")
            conclusion: str = Field(..., description="Overall market assessment")

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
                        "budget_tokens": 1536
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
            response_model=MarketAnalysis,
        )

        response = agent.run("Analyze the electric vehicle market")

        # Verify complex structured output
        assert isinstance(response.content, MarketAnalysis)
        assert response.content.market_name is not None
        assert response.content.market_size > 0
        assert response.content.growth_rate >= 0
        assert len(response.content.key_players) > 0
        assert len(response.content.opportunities) > 0
        assert len(response.content.threats) > 0
        assert response.content.conclusion is not None
        _assert_metrics(response)

    def test_claude_thinking_high_budget_complex_structure(self):
        """Test Claude with high thinking budget for complex structure"""
        class ProjectPlan(BaseModel):
            class Task(BaseModel):
                task_name: str
                description: str
                duration_days: int
                dependencies: List[str]
                assignee: str
                priority: str  # High, Medium, Low

            class Milestone(BaseModel):
                name: str
                date: str
                deliverables: List[str]

            class Risk(BaseModel):
                description: str
                probability: str  # High, Medium, Low
                impact: str  # High, Medium, Low
                mitigation: str

            project_name: str = Field(..., description="Project name")
            duration_weeks: int = Field(..., description="Total project duration in weeks")
            tasks: List[Task] = Field(..., description="List of project tasks")
            milestones: List[Milestone] = Field(..., description="Project milestones")
            risks: List[Risk] = Field(..., description="Project risks")
            budget: float = Field(..., description="Project budget in USD")

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
                        "budget_tokens": 2048  # High budget for complex planning
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
            response_model=ProjectPlan,
        )

        response = agent.run("Create a detailed project plan for developing a mobile app")

        # Verify complex nested structured output
        assert isinstance(response.content, ProjectPlan)
        assert response.content.project_name is not None
        assert response.content.duration_weeks > 0
        assert len(response.content.tasks) > 0
        assert len(response.content.milestones) > 0
        assert len(response.content.risks) > 0
        assert response.content.budget > 0

        # Check nested structures
        for task in response.content.tasks:
            assert task.task_name is not None
            assert task.duration_days > 0
            assert task.priority in ["High", "Medium", "Low"]

        for milestone in response.content.milestones:
            assert milestone.name is not None
            assert len(milestone.deliverables) > 0

        for risk in response.content.risks:
            assert risk.description is not None
            assert risk.probability in ["High", "Medium", "Low"]
            assert risk.impact in ["High", "Medium", "Low"]

        _assert_metrics(response)

    def test_claude_union_types_thinking(self):
        """Test Claude structured response with Union types and thinking"""
        class NumberOrString(BaseModel):
            value: Union[int, float, str] = Field(..., description="A number or string value")
            value_type: str = Field(..., description="Type of the value")

        class FlexibleData(BaseModel):
            primary_field: Union[str, int] = Field(..., description="Primary data field")
            secondary_fields: List[Union[str, int, float]] = Field(..., description="Secondary data fields")
            metadata: Optional[Union[dict, str]] = Field(None, description="Optional metadata")

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
            response_model=FlexibleData,
        )

        response = agent.run("Create a flexible data structure with mixed types for a user profile")

        # Verify Union types work correctly
        assert isinstance(response.content, FlexibleData)
        assert response.content.primary_field is not None
        assert len(response.content.secondary_fields) > 0
        
        # Check that Union types are handled properly
        for field in response.content.secondary_fields:
            assert isinstance(field, (str, int, float))
            
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_claude_async_structured_thinking(self):
        """Test Claude async structured response with thinking"""
        class TravelItinerary(BaseModel):
            destination: str = Field(..., description="Travel destination")
            duration_days: int = Field(..., description="Trip duration in days")
            activities: List[str] = Field(..., description="List of planned activities")
            accommodation: str = Field(..., description="Accommodation details")
            budget: float = Field(..., description="Estimated budget")
            best_season: str = Field(..., description="Best season to visit")

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
            response_model=TravelItinerary,
        )

        response = await agent.arun("Plan a 7-day trip to Japan")

        # Verify structured output
        assert isinstance(response.content, TravelItinerary)
        assert response.content.destination is not None
        assert response.content.duration_days == 7
        assert len(response.content.activities) > 0
        assert response.content.accommodation is not None
        assert response.content.budget > 0
        assert response.content.best_season is not None
        _assert_metrics(response)


class TestStructuredResponseComparison:
    """Test structured response functionality comparing both models"""

    def test_both_models_same_structure_different_reasoning(self):
        """Test both models with same structure but different reasoning approaches"""
        class ProductReview(BaseModel):
            product_name: str = Field(..., description="Name of the product")
            rating: float = Field(..., description="Rating out of 5")
            pros: List[str] = Field(..., description="Positive aspects")
            cons: List[str] = Field(..., description="Negative aspects")
            recommendation: str = Field(..., description="Final recommendation")

        # GPT-5 with reasoning
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
            response_model=ProductReview,
        )

        gpt5_response = gpt5_agent.run("Review the iPhone 15 Pro")

        # Claude with thinking
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
            response_model=ProductReview,
        )

        claude_response = claude_agent.run("Review the iPhone 15 Pro")

        # Both should produce valid structured output
        assert isinstance(gpt5_response.content, ProductReview)
        assert isinstance(claude_response.content, ProductReview)

        # Both should have valid ratings
        assert 1 <= gpt5_response.content.rating <= 5
        assert 1 <= claude_response.content.rating <= 5

        # Both should have pros and cons
        assert len(gpt5_response.content.pros) > 0
        assert len(gpt5_response.content.cons) > 0
        assert len(claude_response.content.pros) > 0
        assert len(claude_response.content.cons) > 0

        _assert_metrics(gpt5_response)
        _assert_metrics(claude_response)

    def test_complex_reasoning_vs_thinking_structured_output(self):
        """Test complex structured output comparing reasoning vs thinking approaches"""
        class FinancialReport(BaseModel):
            class QuarterlyData(BaseModel):
                quarter: str
                revenue: float
                expenses: float
                profit: float

            company_name: str = Field(..., description="Company name")
            year: int = Field(..., description="Report year")
            quarterly_data: List[QuarterlyData] = Field(..., description="Quarterly financial data")
            annual_revenue: float = Field(..., description="Total annual revenue")
            annual_profit: float = Field(..., description="Total annual profit")
            growth_rate: float = Field(..., description="Year over year growth rate")
            key_insights: List[str] = Field(..., description="Key financial insights")

        # GPT-5 with high reasoning effort
        gpt5_agent = Agent(
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
            response_model=FinancialReport,
        )

        gpt5_response = gpt5_agent.run("Generate a financial report for a tech company showing growth")

        # Claude with high thinking budget
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
                        "budget_tokens": 2048
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
            response_model=FinancialReport,
        )

        claude_response = claude_agent.run("Generate a financial report for a tech company showing growth")

        # Verify both produce valid complex structured output
        for response in [gpt5_response, claude_response]:
            assert isinstance(response.content, FinancialReport)
            assert response.content.company_name is not None
            assert response.content.year > 2020
            assert len(response.content.quarterly_data) == 4  # Should have 4 quarters
            assert response.content.annual_revenue > 0
            assert response.content.annual_profit > 0
            assert len(response.content.key_insights) > 0

            # Check quarterly data structure
            for quarter_data in response.content.quarterly_data:
                assert quarter_data.quarter is not None
                assert quarter_data.revenue > 0
                assert quarter_data.profit == quarter_data.revenue - quarter_data.expenses

            _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_async_structured_comparison(self):
        """Test async structured responses from both models"""
        class EventPlan(BaseModel):
            event_name: str = Field(..., description="Name of the event")
            date: str = Field(..., description="Event date")
            venue: str = Field(..., description="Event venue")
            attendees: int = Field(..., description="Expected number of attendees")
            agenda: List[str] = Field(..., description="Event agenda items")
            budget: float = Field(..., description="Event budget")

        # Both models without reasoning/thinking first
        gpt5_simple = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            response_model=EventPlan,
            telemetry=False,
            monitoring=False,
        )

        claude_simple = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            response_model=EventPlan,
            telemetry=False,
            monitoring=False,
        )

        # Run both async
        gpt5_response, claude_response = await pytest.gather(
            gpt5_simple.arun("Plan a tech conference"),
            claude_simple.arun("Plan a tech conference")
        )

        # Both should produce valid structured output
        assert isinstance(gpt5_response.content, EventPlan)
        assert isinstance(claude_response.content, EventPlan)

        # Both should have reasonable values
        for response in [gpt5_response, claude_response]:
            assert response.content.attendees > 0
            assert response.content.budget > 0
            assert len(response.content.agenda) > 0
            _assert_metrics(response)