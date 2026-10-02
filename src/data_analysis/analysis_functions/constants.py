# Research variables and the fixed significance threshold

ALPHA = 0.05

MAINTENANCE_SCORE_COLUMN = "software_maintenance_efficiency_score"

DOCUMENTATION_QUALITY_SCORE_COLUMN = "documentation_quality_score"

DOCUMENTATION_METRIC_COLUMNS = [
    "documentation_completeness_score",
    "installation_guidance_score",
    "usage_guidance_score",
    "configuration_documentation_score",
    "dependency_documentation_score",
    "documentation_readability_score",
    "documentation_clarity_score",
    "documentation_coverage_score",
    "deployment_guidance_score",
    "troubleshooting_support_score",
    "architecture_documentation_score",
    "documentation_consistency_score",
    "example_availability_score",
    "command_availability_score",
    "version_information_score",
]

MAINTENANCE_VARIABLE_COLUMNS = [
    "github_issue_closure_rate",
    "github_avg_issue_resolution_time_score",
    "github_pr_merge_rate",
    "github_avg_issue_comment_count_score",
    MAINTENANCE_SCORE_COLUMN,
]

RESEARCH_TITLE = (
    "The Impact of Technical Documentation Quality on Software Maintenance Efficiency in Enterprise-Level Software Projects: "
    "An Empirical Analysis of Open-Source GitHub Repository Data"
)

RESEARCH_QUESTION = (
    "Does Technical Documentation Quality have a statistically significant relationship with Software Maintenance Efficiency in enterprise-level software projects?"
)

H1 = (
    "Technical Documentation Quality has a statistically significant relationship with Software Maintenance Efficiency in enterprise-level software projects."
)

H0 = (
    "Technical Documentation Quality does not have a statistically significant relationship with Software Maintenance Efficiency in enterprise-level software projects."
)


# Function to convert column names into readable labels
def readable_name(column_name):
    return column_name.replace("_score", "").replace("_", " ").title()
