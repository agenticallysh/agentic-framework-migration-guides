#!/usr/bin/env python3
"""
OpenAI Assistants Cost Calculator
=================================

Calculates potential cost savings from migrating OpenAI Assistants to CrewAI.
Analyzes usage patterns and provides detailed cost projections.
"""

import json
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CostAnalysis:
    """Cost analysis results"""
    current_monthly_cost: float = 0.0
    projected_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    annual_savings: float = 0.0
    savings_percentage: float = 0.0
    payback_period_months: float = 0.0
    migration_cost: float = 0.0

class OpenAICostCalculator:
    """Calculates costs for OpenAI Assistants vs CrewAI alternatives"""
    
    # OpenAI Assistants pricing (as of 2025)
    OPENAI_PRICING = {
        "gpt-4-turbo": {
            "input_tokens": 0.01 / 1000,   # $0.01 per 1K tokens
            "output_tokens": 0.03 / 1000,  # $0.03 per 1K tokens
        },
        "gpt-4": {
            "input_tokens": 0.03 / 1000,
            "output_tokens": 0.06 / 1000,
        },
        "gpt-3.5-turbo": {
            "input_tokens": 0.0005 / 1000,
            "output_tokens": 0.0015 / 1000,
        }
    }
    
    # Assistant management overhead (estimated)
    ASSISTANT_OVERHEAD = 0.25  # 25% overhead for managed service
    
    # File storage costs
    FILE_STORAGE_COST_PER_GB = 20.0  # $20/GB/month
    
    # Alternative LLM pricing
    ALTERNATIVE_PRICING = {
        "openai_direct": {
            "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
            "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
            "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
            "overhead": 0.0  # No managed service overhead
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015/1000, "output": 0.075/1000},
            "claude-3-sonnet": {"input": 0.003/1000, "output": 0.015/1000},
            "claude-3-haiku": {"input": 0.00025/1000, "output": 0.00125/1000},
            "overhead": 0.0
        },
        "local_models": {
            "mixtral-8x7b": {"input": 0.0002/1000, "output": 0.0002/1000},
            "llama-2-70b": {"input": 0.0001/1000, "output": 0.0001/1000},
            "overhead": 0.0,
            "infrastructure_cost": 500  # Monthly server costs
        }
    }
    
    def __init__(self):
        self.usage_data = {}
        
    def analyze_usage_file(self, usage_file: str) -> Dict[str, Any]:
        """Analyze usage data from JSON file"""
        
        with open(usage_file, 'r') as f:
            self.usage_data = json.load(f)
        
        return self._calculate_costs()
    
    def analyze_manual_input(self, 
                           monthly_tokens: int,
                           model: str = "gpt-4-turbo",
                           num_assistants: int = 1,
                           file_storage_gb: float = 0.0) -> Dict[str, Any]:
        """Analyze costs from manual input"""
        
        self.usage_data = {
            "monthly_tokens": monthly_tokens,
            "model": model,
            "num_assistants": num_assistants,
            "file_storage_gb": file_storage_gb,
            "input_output_ratio": 0.6  # Assume 60% input, 40% output
        }
        
        return self._calculate_costs()
    
    def _calculate_costs(self) -> Dict[str, Any]:
        """Calculate current and projected costs"""
        
        monthly_tokens = self.usage_data.get("monthly_tokens", 0)
        model = self.usage_data.get("model", "gpt-4-turbo")
        num_assistants = self.usage_data.get("num_assistants", 1)
        file_storage_gb = self.usage_data.get("file_storage_gb", 0.0)
        input_ratio = self.usage_data.get("input_output_ratio", 0.6)
        
        # Calculate current OpenAI Assistants costs
        current_cost = self._calculate_openai_cost(
            monthly_tokens, model, num_assistants, file_storage_gb, input_ratio
        )
        
        # Calculate alternative costs
        alternatives = {}
        for provider, pricing in self.ALTERNATIVE_PRICING.items():
            alternatives[provider] = self._calculate_alternative_cost(
                monthly_tokens, model, provider, file_storage_gb, input_ratio
            )
        
        # Find best alternative
        best_alternative = min(alternatives.items(), key=lambda x: x[1]["total_cost"])
        
        # Calculate migration costs
        migration_cost = self._estimate_migration_cost(num_assistants)
        
        return {
            "current_cost": current_cost,
            "alternatives": alternatives,
            "best_alternative": {
                "provider": best_alternative[0],
                "costs": best_alternative[1]
            },
            "migration_cost": migration_cost,
            "savings_analysis": self._calculate_savings(
                current_cost["total_cost"],
                best_alternative[1]["total_cost"],
                migration_cost
            )
        }
    
    def _calculate_openai_cost(self, tokens: int, model: str, assistants: int, 
                              storage_gb: float, input_ratio: float) -> Dict[str, float]:
        """Calculate OpenAI Assistants costs"""
        
        if model not in self.OPENAI_PRICING:
            model = "gpt-4-turbo"  # Default fallback
        
        pricing = self.OPENAI_PRICING[model]
        
        input_tokens = int(tokens * input_ratio)
        output_tokens = tokens - input_tokens
        
        # Base token costs
        token_cost = (
            input_tokens * pricing["input_tokens"] + 
            output_tokens * pricing["output_tokens"]
        )
        
        # Assistant management overhead
        overhead_cost = token_cost * self.ASSISTANT_OVERHEAD
        
        # File storage costs
        storage_cost = storage_gb * self.FILE_STORAGE_COST_PER_GB
        
        # Assistant setup/management fees (estimated)
        management_fee = assistants * 50  # $50/assistant/month
        
        total_cost = token_cost + overhead_cost + storage_cost + management_fee
        
        return {
            "token_cost": token_cost,
            "overhead_cost": overhead_cost,
            "storage_cost": storage_cost,
            "management_fee": management_fee,
            "total_cost": total_cost
        }
    
    def _calculate_alternative_cost(self, tokens: int, original_model: str, 
                                  provider: str, storage_gb: float, 
                                  input_ratio: float) -> Dict[str, float]:
        """Calculate costs for alternative providers"""
        
        pricing_data = self.ALTERNATIVE_PRICING[provider]
        
        # Map original model to alternative
        if provider == "openai_direct":
            model_key = original_model
        elif provider == "anthropic":
            # Map to equivalent Anthropic models
            model_mapping = {
                "gpt-4": "claude-3-opus",
                "gpt-4-turbo": "claude-3-sonnet", 
                "gpt-3.5-turbo": "claude-3-haiku"
            }
            model_key = model_mapping.get(original_model, "claude-3-sonnet")
        else:  # local_models
            model_key = "mixtral-8x7b"  # Default local model
        
        # Get pricing for the model
        if model_key in pricing_data:
            model_pricing = pricing_data[model_key]
        else:
            # Use first available model
            model_pricing = next(iter(pricing_data.values()))
        
        input_tokens = int(tokens * input_ratio)
        output_tokens = tokens - input_tokens
        
        # Calculate token costs
        token_cost = (
            input_tokens * model_pricing["input"] + 
            output_tokens * model_pricing["output"]
        )
        
        # Infrastructure costs (for local models)
        infrastructure_cost = pricing_data.get("infrastructure_cost", 0)
        
        # CrewAI is free (open source)
        framework_cost = 0
        
        # File storage (self-managed, much cheaper)
        storage_cost = storage_gb * 2  # $2/GB for self-managed storage
        
        total_cost = token_cost + infrastructure_cost + framework_cost + storage_cost
        
        return {
            "token_cost": token_cost,
            "infrastructure_cost": infrastructure_cost,
            "framework_cost": framework_cost,
            "storage_cost": storage_cost,
            "total_cost": total_cost,
            "model_used": model_key
        }
    
    def _estimate_migration_cost(self, num_assistants: int) -> float:
        """Estimate one-time migration costs"""
        
        # Development time estimate
        base_hours = 8  # Base setup time
        per_assistant_hours = 2  # Hours per assistant to migrate
        hourly_rate = 100  # Developer hourly rate
        
        development_cost = (base_hours + num_assistants * per_assistant_hours) * hourly_rate
        
        # Testing and validation
        testing_cost = development_cost * 0.3
        
        # Documentation and training
        docs_cost = 500
        
        return development_cost + testing_cost + docs_cost
    
    def _calculate_savings(self, current_cost: float, new_cost: float, 
                          migration_cost: float) -> Dict[str, float]:
        """Calculate savings metrics"""
        
        monthly_savings = current_cost - new_cost
        annual_savings = monthly_savings * 12
        savings_percentage = (monthly_savings / current_cost) * 100 if current_cost > 0 else 0
        
        # Payback period in months
        payback_months = migration_cost / monthly_savings if monthly_savings > 0 else float('inf')
        
        return {
            "monthly_savings": monthly_savings,
            "annual_savings": annual_savings,
            "savings_percentage": savings_percentage,
            "payback_period_months": payback_months,
            "three_year_savings": annual_savings * 3 - migration_cost
        }
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a detailed cost analysis report"""
        
        current = analysis["current_cost"]
        best = analysis["best_alternative"]
        migration = analysis["migration_cost"]
        savings = analysis["savings_analysis"]
        
        report = f"""
💰 OpenAI Assistants → CrewAI Cost Analysis
==========================================

📊 Current OpenAI Assistants Costs (Monthly):
  • Token costs: ${current['token_cost']:,.2f}
  • Management overhead: ${current['overhead_cost']:,.2f}
  • Storage costs: ${current['storage_cost']:,.2f}
  • Platform fees: ${current['management_fee']:,.2f}
  • Total monthly: ${current['total_cost']:,.2f}

🚀 Best Alternative ({best['provider'].title()}):
  • Model: {best['costs'].get('model_used', 'N/A')}
  • Token costs: ${best['costs']['token_cost']:,.2f}
  • Infrastructure: ${best['costs']['infrastructure_cost']:,.2f}
  • Storage: ${best['costs']['storage_cost']:,.2f}
  • Framework: ${best['costs']['framework_cost']:,.2f} (CrewAI is free!)
  • Total monthly: ${best['costs']['total_cost']:,.2f}

💡 Migration Investment:
  • One-time migration cost: ${migration:,.2f}
  • Estimated timeline: 2-4 weeks

📈 Savings Projection:
  • Monthly savings: ${savings['monthly_savings']:,.2f}
  • Annual savings: ${savings['annual_savings']:,.2f}
  • Savings percentage: {savings['savings_percentage']:.1f}%
  • Payback period: {savings['payback_period_months']:.1f} months
  • 3-year net savings: ${savings['three_year_savings']:,.2f}

🎯 Other Provider Options:
"""
        
        for provider, costs in analysis["alternatives"].items():
            if provider != best['provider']:
                monthly_savings_alt = current['total_cost'] - costs['total_cost']
                savings_pct = (monthly_savings_alt / current['total_cost']) * 100
                report += f"  • {provider.title()}: ${costs['total_cost']:,.2f}/month ({savings_pct:.1f}% savings)\n"
        
        report += f"""
🚨 Key Considerations:
  • Migration requires 2-4 weeks of development time
  • CrewAI provides better agent coordination than Assistants API
  • Full control over infrastructure and data
  • No vendor lock-in with open source solution
  • Can switch between multiple LLM providers

📞 Next Steps:
  1. Review migration guide: README.md
  2. Start with pilot assistant migration
  3. Plan infrastructure if using local models
  4. Schedule team training on CrewAI

🔗 Resources:
  • Migration guide: https://www.agentically.sh/ai-agentic-frameworks/migrate/openai-assistants-to-crewai/
  • CrewAI documentation: https://docs.crewai.com/
  • Professional migration services: https://www.agentically.sh/migration-consulting/
"""
        
        return report

def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Calculate cost savings from migrating OpenAI Assistants to CrewAI"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--usage-file", "-f",
        help="JSON file with usage data"
    )
    group.add_argument(
        "--manual", "-m",
        action="store_true",
        help="Enter usage data manually"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results (optional)"
    )
    
    args = parser.parse_args()
    
    calculator = OpenAICostCalculator()
    
    if args.usage_file:
        # Analyze from file
        try:
            analysis = calculator.analyze_usage_file(args.usage_file)
        except FileNotFoundError:
            print(f"❌ Error: Usage file '{args.usage_file}' not found")
            return
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in '{args.usage_file}'")
            return
    
    else:
        # Manual input
        print("📝 Manual Cost Analysis")
        print("=" * 30)
        
        try:
            monthly_tokens = int(input("Monthly token usage: "))
            
            print("\nAvailable models:")
            for i, model in enumerate(calculator.OPENAI_PRICING.keys(), 1):
                print(f"  {i}. {model}")
            
            model_choice = input("\nSelect model (1-3, default 1): ") or "1"
            models = list(calculator.OPENAI_PRICING.keys())
            model = models[int(model_choice) - 1]
            
            num_assistants = int(input("Number of assistants: ") or "1")
            file_storage = float(input("File storage (GB, default 0): ") or "0")
            
            analysis = calculator.analyze_manual_input(
                monthly_tokens, model, num_assistants, file_storage
            )
            
        except (ValueError, IndexError):
            print("❌ Error: Invalid input provided")
            return
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled")
            return
    
    # Generate and display report
    report = calculator.generate_report(analysis)
    print(report)
    
    # Save JSON output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main()

"""
Usage Examples:
===============

# Manual analysis
python cost_calculator.py --manual

# Analyze from usage file
python cost_calculator.py --usage-file usage_data.json

# Save detailed results
python cost_calculator.py --manual --output analysis.json

Sample usage_data.json:
{
    "monthly_tokens": 1000000,
    "model": "gpt-4-turbo",
    "num_assistants": 5,
    "file_storage_gb": 10.0,
    "input_output_ratio": 0.6
}

This tool helps you:
1. Calculate exact cost savings potential
2. Compare different LLM provider options
3. Estimate migration investment required
4. Plan your migration timeline and budget
"""