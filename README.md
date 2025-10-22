# Framework Migration Guides

🔄 Comprehensive guides for migrating between AI agent frameworks. Save weeks of refactoring with step-by-step migrations, code mappings, and gotchas.

[![Migration Guides](https://img.shields.io/badge/Guides-12+-blue.svg)](https://github.com/agenticallysh/agentic-framework-migration-guides)
[![Success Rate](https://img.shields.io/badge/Success%20Rate-94%25-green.svg)](https://www.agentically.sh/ai-agentic-frameworks/migration-success/)
[![Time Saved](https://img.shields.io/badge/Time%20Saved-60%25-orange.svg)](https://www.agentically.sh/ai-agentic-frameworks/migration-calculator/)

## 🎯 Quick Migration Finder

Not sure which migration path to take? Use our [migration assistant →](https://www.agentically.sh/ai-agentic-frameworks/migration-tool/)

## 🔥 Most Popular Migrations

| From → To | Complexity | Time | Success Rate | Guide |
|-----------|------------|------|-------------|--------|
| [LangChain → CrewAI](./migrations/langchain-to-crewai/) | Medium | 3-5 hours | 96% | [Start →](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-crewai/) |
| [AutoGen → LangGraph](./migrations/autogen-to-langgraph/) | High | 6-8 hours | 89% | [Start →](https://www.agentically.sh/ai-agentic-frameworks/migrate/autogen-to-langgraph/) |
| [OpenAI → AgentKit](./migrations/openai-to-agentkit/) | Low | 1-2 hours | 98% | [Start →](https://www.agentically.sh/ai-agentic-frameworks/migrate/openai-to-agentkit/) |
| [LangChain → AutoGen](./migrations/langchain-to-autogen/) | High | 8-12 hours | 87% | [Start →](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-autogen/) |

[View all migration paths →](https://www.agentically.sh/ai-agentic-frameworks/migration-paths/)

## 📊 Migration Calculator

Estimate time and effort for your specific migration:
[Interactive Migration Calculator →](https://www.agentically.sh/ai-agentic-frameworks/migration-calculator/)

## 🛠️ Available Migrations

### From LangChain
- [LangChain → CrewAI](./migrations/langchain-to-crewai/) - Better multi-agent coordination
- [LangChain → AutoGen](./migrations/langchain-to-autogen/) - Advanced conversation patterns
- [LangChain → LangGraph](./migrations/langchain-to-langgraph/) - Stateful workflows
- [LangChain → Semantic Kernel](./migrations/langchain-to-semantic-kernel/) - Enterprise integration

### From AutoGen
- [AutoGen → CrewAI](./migrations/autogen-to-crewai/) - Simplified multi-agent setup
- [AutoGen → LangGraph](./migrations/autogen-to-langgraph/) - Graph-based workflows
- [AutoGen → Agency Swarm](./migrations/autogen-to-agency-swarm/) - Scalable agent networks

### From CrewAI
- [CrewAI → AutoGen](./migrations/crewai-to-autogen/) - More complex conversations
- [CrewAI → LangGraph](./migrations/crewai-to-langgraph/) - Advanced state management

### Platform Migrations
- [OpenAI Assistants → Various Frameworks](./migrations/openai-assistants/)
- [Google Vertex AI → Open Source](./migrations/vertex-to-opensource/)
- [Azure OpenAI → Self-hosted](./migrations/azure-to-selfhosted/)

## 🚀 Migration Tools

### Automated Migration Assistant
```bash
pip install agentically-migration-tools
agentically-migrate --from langchain --to crewai --project ./my-project
```

### Code Analysis Tool
```bash
agentically-analyze --framework detect --path ./
# Outputs: Detected LangChain v0.1.0 with 5 agents, 12 tools
```

### Compatibility Checker
```bash
agentically-compat --current langchain --target crewai
# Outputs: 89% compatible, 3 breaking changes found
```

[Get migration tools →](https://www.agentically.sh/ai-agentic-frameworks/migration-tools/)

## 📈 Migration Success Stories

> *"Migrated our 15-agent LangChain system to CrewAI in 2 days. 40% performance improvement and much cleaner code."*  
> — **TechCorp Engineering Team**

> *"The AutoGen to LangGraph migration guide saved us 3 weeks of development time."*  
> — **Sarah Chen, AI Lead at DataFlow**

[Read more success stories →](https://www.agentically.sh/ai-agentic-frameworks/migration-stories/)

## 🎓 Migration Best Practices

### Before You Start
1. **Audit Current System**: Use our analysis tools
2. **Set Clear Goals**: Performance, maintainability, features?
3. **Plan Downtime**: Most migrations require 2-8 hours
4. **Backup Everything**: Create migration branch

### During Migration
1. **Follow Guide Step-by-Step**: Don't skip steps
2. **Test Incrementally**: Verify each component
3. **Monitor Performance**: Benchmark before/after
4. **Document Changes**: Track what was modified

### After Migration
1. **Performance Testing**: Load test the new system
2. **Team Training**: Ensure everyone understands changes
3. **Update Documentation**: Reflect new architecture
4. **Plan Optimization**: Leverage new framework features

[Complete migration checklist →](https://www.agentically.sh/ai-agentic-frameworks/migration-checklist/)

## 🧪 Testing Your Migration

### Automated Testing Suite
```python
from agentically_migration_tools import MigrationTester

tester = MigrationTester()
results = tester.compare_outputs(
    original_framework="langchain",
    migrated_framework="crewai",
    test_cases=["test1.json", "test2.json"]
)

print(f"Accuracy: {results.accuracy}%")
print(f"Performance: {results.performance_delta}%")
```

### Manual Testing Checklist
- [ ] All agents respond correctly
- [ ] Tools function as expected
- [ ] Memory/context preserved
- [ ] Error handling works
- [ ] Performance meets requirements

## 💰 Cost Impact Analysis

Most migrations result in cost savings:

| Migration | Avg Cost Change | Token Efficiency | Performance |
|-----------|----------------|------------------|-------------|
| LangChain → CrewAI | -23% | +15% | +18% |
| AutoGen → LangGraph | -12% | +8% | +25% |
| Any → OpenSource | -67% | Varies | Varies |

[Calculate your savings →](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/)

## 🚨 Common Migration Pitfalls

### 1. Underestimating Complexity
- **Problem**: "It's just changing imports, right?"
- **Reality**: Architecture differences require careful planning
- **Solution**: Use our complexity calculator first

### 2. Skipping Testing
- **Problem**: Assuming equivalent functionality
- **Reality**: Subtle behavior differences can break workflows
- **Solution**: Comprehensive testing at each step

### 3. Not Optimizing Post-Migration
- **Problem**: Direct port without leveraging new features
- **Reality**: Missing 30-50% of potential benefits
- **Solution**: Follow our optimization guides

[Complete pitfalls guide →](https://www.agentically.sh/ai-agentic-frameworks/migration-pitfalls/)

## 🤝 Migration Support

### Community Support
- [Discord #migrations channel](https://discord.gg/agentically)
- [GitHub Discussions](https://github.com/agenticallysh/agentic-framework-migration-guides/discussions)
- [Weekly Migration Office Hours](https://www.agentically.sh/ai-agentic-frameworks/office-hours/)

### Professional Services
- [Migration Consulting](https://www.agentically.sh/ai-agentic-frameworks/migration-consulting/) - Expert guidance
- [Code Review](https://www.agentically.sh/ai-agentic-frameworks/migration-review/) - Post-migration optimization
- [Team Training](https://www.agentically.sh/ai-agentic-frameworks/training/) - Framework-specific workshops

## 📚 Additional Resources

- [Framework Comparison Tool](https://www.agentically.sh/ai-agentic-frameworks/compare/) - Choose your target
- [Performance Benchmarks](https://www.agentically.sh/ai-agentic-frameworks/benchmarks/) - Expected improvements
- [Cost Calculator](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/) - Financial impact
- [Architecture Patterns](https://www.agentically.sh/ai-agentic-frameworks/patterns/) - Best practices

## 🔗 Quick Links

- [All AI Agent Frameworks →](https://github.com/agenticallysh/ai-agentic-frameworks)
- [Performance Benchmarks →](https://github.com/agenticallysh/agent-framework-benchmarks)
- [Production Templates →](https://github.com/agenticallysh/production-agent-templates)
- [Weekly Updates →](https://github.com/agenticallysh/weekly-agent-updates)

---

Built with ❤️ by [Agentically](https://www.agentically.sh) | [Start Your Migration →](https://www.agentically.sh/ai-agentic-frameworks/migration-tool/)