#!/usr/bin/env python3
"""
GitHub Repository Network Analysis

This example demonstrates Neo4jAlchemy's power by analyzing real GitHub data:
- Repository relationships and dependencies
- Developer collaboration networks  
- Programming language ecosystems
- Influence scoring and community detection

Data Source: GitHub API (public repositories)
Requirements: requests, pandas, matplotlib (optional)
"""

import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Import our powerful Neo4jAlchemy core
from neo4jalchemy.core.graph import Graph, GraphNode, GraphEdge


class GitHubAnalyzer:
    """
    Real-world GitHub repository network analyzer using Neo4jAlchemy.
    
    This class demonstrates how Neo4jAlchemy's rich data structures
    handle complex real-world data better than traditional graph libraries.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for higher rate limits."""
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({"Authorization": f"token {github_token}"})
        
        # Create our main graph
        self.graph = Graph(name="github_ecosystem")
        
        # Track API rate limiting
        self.api_calls_made = 0
        self.last_api_call = None
    
    def _api_call(self, url: str) -> Dict[str, Any]:
        """Make GitHub API call with rate limiting."""
        if self.last_api_call:
            # Respect rate limiting (1 second between calls for demo)
            time_since_last = time.time() - self.last_api_call
            if time_since_last < 1.0:
                time.sleep(1.0 - time_since_last)
        
        try:
            response = self.session.get(url)
            self.api_calls_made += 1
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error {response.status_code}: {url}")
                return {}
                
        except Exception as e:
            print(f"Request failed: {e}")
            return {}
    
    def fetch_popular_repos(self, language: str = "python", count: int = 20) -> List[Dict]:
        """
        Fetch popular repositories for analysis.
        
        In a real application, you might fetch from multiple sources:
        - GitHub trending repositories
        - Package manager data (PyPI, npm, etc.)
        - Company-specific repositories
        """
        print(f"🔍 Fetching top {count} {language} repositories...")
        
        # GitHub search API for popular repos
        search_url = f"https://api.github.com/search/repositories"
        params = {
            "q": f"language:{language}",
            "sort": "stars",
            "order": "desc",
            "per_page": min(count, 100)
        }
        
        response = self._api_call(f"{search_url}?" + "&".join(f"{k}={v}" for k, v in params.items()))
        
        if "items" in response:
            repos = response["items"][:count]
            print(f"✅ Fetched {len(repos)} repositories")
            return repos
        else:
            print("❌ Failed to fetch repositories")
            return []
    
    def fetch_repo_details(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Fetch detailed information about a repository.
        
        This demonstrates Neo4jAlchemy's ability to handle rich,
        nested data structures that would be difficult in NetworkX.
        """
        repo_name = repo_data["full_name"]
        print(f"📊 Analyzing repository: {repo_name}")
        
        # Get repository details
        repo_url = f"https://api.github.com/repos/{repo_name}"
        repo_details = self._api_call(repo_url)
        
        # Get contributors
        contributors_url = f"https://api.github.com/repos/{repo_name}/contributors"
        contributors = self._api_call(contributors_url)
        if not isinstance(contributors, list):
            contributors = []
        
        # Get programming languages
        languages_url = f"https://api.github.com/repos/{repo_name}/languages"
        languages = self._api_call(languages_url)
        
        # Build rich repository data structure
        rich_repo_data = {
            "basic_info": {
                "name": repo_details.get("name", ""),
                "full_name": repo_details.get("full_name", ""),
                "description": repo_details.get("description", ""),
                "homepage": repo_details.get("homepage", ""),
                "created_at": repo_details.get("created_at", ""),
                "updated_at": repo_details.get("updated_at", ""),
            },
            "metrics": {
                "stars": repo_details.get("stargazers_count", 0),
                "forks": repo_details.get("forks_count", 0),
                "watchers": repo_details.get("watchers_count", 0),
                "open_issues": repo_details.get("open_issues_count", 0),
                "size_kb": repo_details.get("size", 0),
            },
            "activity": {
                "contributors_count": len(contributors),
                "top_contributors": [
                    {
                        "login": contrib.get("login", ""),
                        "contributions": contrib.get("contributions", 0),
                        "avatar_url": contrib.get("avatar_url", "")
                    }
                    for contrib in contributors[:10]  # Top 10 contributors
                ],
                "last_push": repo_details.get("pushed_at", ""),
            },
            "technical": {
                "primary_language": repo_details.get("language", ""),
                "languages": languages,
                "default_branch": repo_details.get("default_branch", "main"),
                "license": repo_details.get("license", {}).get("name", "") if repo_details.get("license") else "",
            },
            "social": {
                "owner": {
                    "login": repo_details.get("owner", {}).get("login", ""),
                    "type": repo_details.get("owner", {}).get("type", ""),
                    "avatar_url": repo_details.get("owner", {}).get("avatar_url", ""),
                },
                "topics": repo_details.get("topics", []),
                "has_wiki": repo_details.get("has_wiki", False),
                "has_pages": repo_details.get("has_pages", False),
            }
        }
        
        return rich_repo_data
    
    def build_repository_network(self, repo_list: List[Dict]):
        """
        Build the repository network graph.
        
        This showcases Neo4jAlchemy's superior data modeling compared to NetworkX:
        - Rich nested properties on nodes and edges
        - Type validation and serialization
        - Business logic integration
        """
        print(f"🏗️ Building repository network graph...")
        
        # Track processed entities
        processed_repos = set()
        processed_users = set()
        
        for repo_basic in repo_list:
            repo_name = repo_basic["full_name"]
            
            # Fetch detailed repo data
            repo_data = self.fetch_repo_details(repo_basic)
            
            # Add repository node with rich data
            repo_node = self.graph.add_node(
                f"repo:{repo_name}",
                "Repository",
                repo_data
            )
            processed_repos.add(repo_name)
            
            # Add owner node
            owner_login = repo_data["social"]["owner"]["login"]
            owner_node_id = f"user:{owner_login}"
            
            if owner_node_id not in processed_users:
                owner_node = self.graph.add_node(
                    owner_node_id,
                    "GitHubUser",
                    {
                        "profile": {
                            "login": owner_login,
                            "type": repo_data["social"]["owner"]["type"],
                            "avatar_url": repo_data["social"]["owner"]["avatar_url"],
                        },
                        "ownership": {
                            "repositories_owned": 1,
                            "total_stars_owned": repo_data["metrics"]["stars"],
                        }
                    }
                )
                processed_users.add(owner_login)
            else:
                # Update existing owner stats
                owner_node = self.graph.get_node(owner_node_id)
                owner_node.properties["ownership"]["repositories_owned"] += 1
                owner_node.properties["ownership"]["total_stars_owned"] += repo_data["metrics"]["stars"]
            
            # Add ownership relationship
            ownership_edge = self.graph.add_edge(
                owner_node_id,
                f"repo:{repo_name}",
                "OWNS",
                {
                    "relationship_type": "ownership",
                    "created_at": repo_data["basic_info"]["created_at"],
                    "repository_metrics": {
                        "stars": repo_data["metrics"]["stars"],
                        "forks": repo_data["metrics"]["forks"],
                    }
                }
            )
            
            # Add contributor relationships
            for contributor in repo_data["activity"]["top_contributors"]:
                contrib_login = contributor["login"]
                contrib_node_id = f"user:{contrib_login}"
                
                # Skip if same as owner
                if contrib_login == owner_login:
                    continue
                
                # Add contributor node if not exists
                if contrib_node_id not in processed_users:
                    contrib_node = self.graph.add_node(
                        contrib_node_id,
                        "GitHubUser",
                        {
                            "profile": {
                                "login": contrib_login,
                                "type": "User",  # Assume regular user
                                "avatar_url": contributor["avatar_url"],
                            },
                            "contributions": {
                                "repositories_contributed": 1,
                                "total_contributions": contributor["contributions"],
                            }
                        }
                    )
                    processed_users.add(contrib_login)
                else:
                    # Update existing contributor stats
                    contrib_node = self.graph.get_node(contrib_node_id)
                    if "contributions" not in contrib_node.properties:
                        contrib_node.properties["contributions"] = {
                            "repositories_contributed": 0,
                            "total_contributions": 0
                        }
                    contrib_node.properties["contributions"]["repositories_contributed"] += 1
                    contrib_node.properties["contributions"]["total_contributions"] += contributor["contributions"]
                
                # Add contribution relationship
                contribution_edge = self.graph.add_edge(
                    contrib_node_id,
                    f"repo:{repo_name}",
                    "CONTRIBUTES_TO",
                    {
                        "contribution_details": {
                            "contributions": contributor["contributions"],
                            "contribution_rank": repo_data["activity"]["top_contributors"].index(contributor) + 1,
                        },
                        "collaboration_strength": min(contributor["contributions"] / 100, 1.0),  # Normalize to 0-1
                    },
                    weight=min(contributor["contributions"] / 50, 1.0)  # Edge weight based on contributions
                )
            
            # Add language relationships
            primary_language = repo_data["technical"]["primary_language"]
            if primary_language:
                lang_node_id = f"lang:{primary_language}"
                
                # Add language node if not exists
                if not self.graph.has_node(lang_node_id):
                    lang_node = self.graph.add_node(
                        lang_node_id,
                        "ProgrammingLanguage",
                        {
                            "language_info": {
                                "name": primary_language,
                                "repositories_using": 1,
                                "total_stars": repo_data["metrics"]["stars"],
                            }
                        }
                    )
                else:
                    # Update language stats
                    lang_node = self.graph.get_node(lang_node_id)
                    lang_node.properties["language_info"]["repositories_using"] += 1
                    lang_node.properties["language_info"]["total_stars"] += repo_data["metrics"]["stars"]
                
                # Add language usage relationship
                usage_edge = self.graph.add_edge(
                    f"repo:{repo_name}",
                    lang_node_id,
                    "WRITTEN_IN",
                    {
                        "usage_details": {
                            "is_primary_language": True,
                            "repository_size": repo_data["metrics"]["size_kb"],
                        }
                    }
                )
        
        print(f"✅ Built network: {self.graph.node_count()} nodes, {self.graph.edge_count()} edges")
        return self.graph
    
    def analyze_ecosystem(self) -> Dict[str, Any]:
        """
        Perform comprehensive ecosystem analysis.
        
        This demonstrates Neo4jAlchemy's analytical capabilities
        that go far beyond basic NetworkX functionality.
        """
        print(f"📈 Analyzing GitHub ecosystem...")
        
        analysis_results = {
            "network_stats": {
                "total_nodes": self.graph.node_count(),
                "total_edges": self.graph.edge_count(),
                "density": self.graph.density(),
                "average_degree": self.graph.average_degree(),
            },
            "repository_analysis": {},
            "developer_analysis": {},
            "language_analysis": {},
            "collaboration_patterns": {},
        }
        
        # Repository Analysis
        repos = [node for node in self.graph._nodes.values() if node.label == "Repository"]
        if repos:
            stars_list = [repo.properties["metrics"]["stars"] for repo in repos]
            forks_list = [repo.properties["metrics"]["forks"] for repo in repos]
            
            analysis_results["repository_analysis"] = {
                "total_repositories": len(repos),
                "total_stars": sum(stars_list),
                "average_stars": sum(stars_list) / len(stars_list),
                "total_forks": sum(forks_list),
                "average_forks": sum(forks_list) / len(forks_list),
                "most_starred": max(repos, key=lambda r: r.properties["metrics"]["stars"]),
                "most_forked": max(repos, key=lambda r: r.properties["metrics"]["forks"]),
            }
        
        # Developer Analysis
        users = [node for node in self.graph._nodes.values() if node.label == "GitHubUser"]
        if users:
            # Calculate developer influence using centrality
            developer_influence = {}
            for user in users:
                centrality = self.graph.degree_centrality(user.id)
                total_contribs = user.properties.get("contributions", {}).get("total_contributions", 0)
                influence_score = centrality * (1 + total_contribs / 1000)  # Weight by contributions
                developer_influence[user.id] = {
                    "centrality": centrality,
                    "total_contributions": total_contribs,
                    "influence_score": influence_score,
                    "login": user.properties["profile"]["login"]
                }
            
            # Find top influencers
            top_influencers = sorted(
                developer_influence.items(), 
                key=lambda x: x[1]["influence_score"], 
                reverse=True
            )[:10]
            
            analysis_results["developer_analysis"] = {
                "total_developers": len(users),
                "top_influencers": [
                    {
                        "login": influencer[1]["login"],
                        "influence_score": influencer[1]["influence_score"],
                        "centrality": influencer[1]["centrality"],
                        "contributions": influencer[1]["total_contributions"]
                    }
                    for influencer in top_influencers
                ]
            }
        
        # Language Analysis
        languages = [node for node in self.graph._nodes.values() if node.label == "ProgrammingLanguage"]
        if languages:
            lang_stats = []
            for lang in languages:
                lang_info = lang.properties["language_info"]
                lang_stats.append({
                    "name": lang_info["name"],
                    "repositories": lang_info["repositories_using"],
                    "total_stars": lang_info["total_stars"],
                    "average_stars_per_repo": lang_info["total_stars"] / lang_info["repositories_using"]
                })
            
            # Sort by popularity (total stars)
            lang_stats.sort(key=lambda x: x["total_stars"], reverse=True)
            
            analysis_results["language_analysis"] = {
                "total_languages": len(languages),
                "language_popularity": lang_stats
            }
        
        # Collaboration Patterns
        contrib_edges = [
            edge for edge in self.graph._edges.values() 
            if edge.relationship_type == "CONTRIBUTES_TO"
        ]
        
        if contrib_edges:
            collaboration_strengths = [
                edge.properties["collaboration_strength"] 
                for edge in contrib_edges
            ]
            
            analysis_results["collaboration_patterns"] = {
                "total_collaborations": len(contrib_edges),
                "average_collaboration_strength": sum(collaboration_strengths) / len(collaboration_strengths),
                "strong_collaborations": len([s for s in collaboration_strengths if s > 0.7]),
            }
        
        return analysis_results
    
    def export_results(self, analysis: Dict[str, Any]):
        """
        Export analysis results to multiple formats.
        
        This demonstrates Neo4jAlchemy's excellent integration
        with data science and business intelligence tools.
        """
        print(f"📤 Exporting analysis results...")
        
        # 1. Export graph to JSON
        graph_json = self.graph.to_json(indent=2)
        with open("github_network.json", "w") as f:
            f.write(graph_json)
        print("✅ Exported graph to github_network.json")
        
        # 2. Export to Pandas DataFrames
        graph_dict = self.graph.to_dict()
        
        # Nodes DataFrame
        nodes_df = pd.DataFrame(graph_dict["nodes"])
        nodes_df.to_csv("github_nodes.csv", index=False)
        print("✅ Exported nodes to github_nodes.csv")
        
        # Edges DataFrame
        edges_df = pd.DataFrame(graph_dict["edges"])
        edges_df.to_csv("github_edges.csv", index=False)
        print("✅ Exported edges to github_edges.csv")
        
        # 3. Export analysis summary
        with open("github_analysis.json", "w") as f:
            # Make analysis JSON serializable
            serializable_analysis = json.loads(json.dumps(analysis, default=str))
            json.dump(serializable_analysis, f, indent=2)
        print("✅ Exported analysis to github_analysis.json")
        
        # 4. Generate summary report
        self._generate_summary_report(analysis)
    
    def _generate_summary_report(self, analysis: Dict[str, Any]):
        """Generate a human-readable summary report."""
        report = f"""
# GitHub Ecosystem Analysis Report
Generated: {datetime.now().isoformat()}

## Network Overview
- **Total Nodes**: {analysis['network_stats']['total_nodes']:,}
- **Total Edges**: {analysis['network_stats']['total_edges']:,}
- **Network Density**: {analysis['network_stats']['density']:.4f}
- **Average Degree**: {analysis['network_stats']['average_degree']:.2f}

## Repository Insights
"""
        
        if "repository_analysis" in analysis and analysis["repository_analysis"]:
            repo_analysis = analysis["repository_analysis"]
            report += f"""
- **Total Repositories**: {repo_analysis['total_repositories']}
- **Total Stars**: {repo_analysis['total_stars']:,}
- **Average Stars per Repo**: {repo_analysis['average_stars']:.1f}
- **Total Forks**: {repo_analysis['total_forks']:,}
"""
        
        if "developer_analysis" in analysis and analysis["developer_analysis"]:
            dev_analysis = analysis["developer_analysis"]
            report += f"""
## Developer Insights
- **Total Developers**: {dev_analysis['total_developers']}

### Top Influencers:
"""
            for i, influencer in enumerate(dev_analysis["top_influencers"][:5], 1):
                report += f"{i}. **{influencer['login']}** - Influence: {influencer['influence_score']:.3f}\n"
        
        if "language_analysis" in analysis and analysis["language_analysis"]:
            lang_analysis = analysis["language_analysis"]
            report += f"""
## Language Ecosystem
- **Total Languages**: {lang_analysis['total_languages']}

### Language Popularity (by total stars):
"""
            for i, lang in enumerate(lang_analysis["language_popularity"][:5], 1):
                report += f"{i}. **{lang['name']}** - {lang['total_stars']:,} stars across {lang['repositories']} repos\n"
        
        # Save report
        with open("github_ecosystem_report.md", "w") as f:
            f.write(report)
        print("✅ Generated summary report: github_ecosystem_report.md")


def main():
    """
    Main analysis pipeline demonstrating Neo4jAlchemy's capabilities.
    """
    print("🚀 GitHub Ecosystem Analysis with Neo4jAlchemy")
    print("=" * 60)
    
    # Initialize analyzer
    # Note: Add your GitHub token for higher rate limits
    # analyzer = GitHubAnalyzer(github_token="your_token_here")
    analyzer = GitHubAnalyzer()
    
    try:
        # Step 1: Fetch repository data
        print("\n📥 STEP 1: Fetching Repository Data")
        repos = analyzer.fetch_popular_repos(language="python", count=10)
        
        if not repos:
            print("❌ No repositories fetched. Check your internet connection.")
            return
        
        # Step 2: Build network graph
        print("\n🏗️ STEP 2: Building Network Graph")
        graph = analyzer.build_repository_network(repos)
        
        # Step 3: Analyze ecosystem
        print("\n📊 STEP 3: Analyzing Ecosystem")
        analysis = analyzer.analyze_ecosystem()
        
        # Step 4: Export results
        print("\n📤 STEP 4: Exporting Results")
        analyzer.export_results(analysis)
        
        # Step 5: Display summary
        print("\n📋 STEP 5: Analysis Summary")
        print("=" * 40)
        print(f"🏗️ Network Built:")
        print(f"   Nodes: {analysis['network_stats']['total_nodes']}")
        print(f"   Edges: {analysis['network_stats']['total_edges']}")
        print(f"   Density: {analysis['network_stats']['density']:.4f}")
        
        if "repository_analysis" in analysis:
            repo_stats = analysis["repository_analysis"]
            print(f"\n📚 Repository Analysis:")
            print(f"   Total Stars: {repo_stats.get('total_stars', 0):,}")
            print(f"   Avg Stars: {repo_stats.get('average_stars', 0):.1f}")
        
        if "developer_analysis" in analysis:
            dev_stats = analysis["developer_analysis"]
            print(f"\n👥 Developer Analysis:")
            print(f"   Total Developers: {dev_stats['total_developers']}")
            if dev_stats["top_influencers"]:
                top_dev = dev_stats["top_influencers"][0]
                print(f"   Top Influencer: {top_dev['login']} (score: {top_dev['influence_score']:.3f})")
        
        print(f"\n✅ Analysis complete! Check the exported files:")
        print(f"   📄 github_network.json - Full graph data")
        print(f"   📊 github_nodes.csv - Nodes for analysis")
        print(f"   📊 github_edges.csv - Edges for analysis")
        print(f"   📋 github_ecosystem_report.md - Summary report")
        
        print(f"\n🎉 This demonstrates Neo4jAlchemy's power:")
        print(f"   ✅ Rich data modeling (complex nested properties)")
        print(f"   ✅ Type-safe operations (Pydantic validation)")
        print(f"   ✅ Graph algorithms (centrality, analysis)")
        print(f"   ✅ Data integration (JSON, CSV, pandas)")
        print(f"   ✅ Real-world applicability (GitHub ecosystem)")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()