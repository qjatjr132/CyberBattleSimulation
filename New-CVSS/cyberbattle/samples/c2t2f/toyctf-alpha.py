# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model a toy Capture the flag exercise

See Jupyter notebook toyctf-simulation.ipynb for an example of
game played on this simulation.
"""

# Node1 = Website, Node2 = Website.Directory, Node3 = GitHubProject, Node4 = Sharepoint, Node5 = Website[user=monitor], Node6 = AzureResourceManager, Node7 = AzureResourceManager[user=monitor], Node8 = AzureStorage, Node9 = AzureVM
# Node2Cred: GitExposedToken (GIT 레포지토리에 대한 노출된 접근 토큰)
# Node3Cred: StrongSSHKey
# Node4Cred: SharepointAccess (Sharepoint에 접근하기 위한 권한)
# Node5Cred: WebMonitorCredential (웹 모니터링에 사용되는 크리덴셜을 의미함)
# Node6Cred: ResourceManagerAccess
# Node8Cred: BlobStorageToken

#노드 값 재정의 (서비스 중요도, 방화벽 구성, 속성 중요도, 보안 취약점에 따라 재정의함) - Base Value: 100 -> BASE Value + 재정의 값 = 새로운 정의값
#Website Node: 35 / Website.Directory Node: 60 / GitHubProject Node: 50 / Sharepoint Node: 55 / Website[user=monitor] Node: 30 / AzureResourceManahger Node: 40 / AzureResourceManahger[user=monitor] Node: 45 / AzureStorage Node: 40 / AzureVM Node: 40


from cyberbattle.simulation import model as m
from cyberbattle.simulation.model import NodeID, NodeInfo, VulnerabilityID, VulnerabilityInfo
from typing import Dict, Iterator, cast, Tuple

weight = 0.3
weight_2 = 0.03
default_allow_rules = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)]

# Network nodes involved in the Capture the flag game
nodes = {
    "Website": m.NodeInfo(
        services=[m.ListeningService("HTTPS"),
                  m.ListeningService("SSH")],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW), m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value = 135 * weight_2,
        properties=["MySql", "Ubuntu", "nginx/1.10.3"],
        owned_string="FLAG: Login using insecure SSH user/password",
        vulnerabilities=dict(
            #웹 사이트의 내용을 스캔하여 민감한 정보를 찾아내는 것 (흔히 발생하는 취약점)
            ScanPageContent=m.VulnerabilityInfo(
                description="LeakedGitHubProjectUrl: Website page content shows a link to GitHub ",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["Website.Directory"]),
                reward_string="WEBSITE page content has a link to github -> Github project discovered!",
                cost=1.5 * weight
            ),
            #웹 페이지 소스 코드 내에 민감한 정보가 포함되어 있는 경우
            ScanPageSource=m.VulnerabilityInfo(
                description="Website page source contains refrence to browseable "
                            "relative web directory",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["GitHubProject"]),
                reward_string="Viewing the web page source reveals a URL to a .txt file and directory on the website",
                cost=1.5 * weight
            ),
            #강력한 인증 요구는 보안 강화와 관련이 있으나, 우회할 수 있는 경우를 나타냄
            StrongAuthRequirement=m.VulnerabilityInfo(
                description="Requires strong authentication for SSH access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="Sharepoint", port="SSH", credential="SharepointAccess")
                ]),
                reward_string="Required strong SSH key for authentication, enhancing security.",
                cost=1.5 * weight
            ))),

    "Website.Directory": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS"),
            m.ListeningService("GIT", allowedCredentials=["GitExposedToken"])
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("GIT", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=160 * weight_2,
        properties=["GitHub", "SasUrlInCommit"],
        vulnerabilities=dict(
            #Git 기록이 노출되어 이전 커밋에서 민감한 정보를 찾아낼 수 있는 경
            ExposedGitHistory=m.VulnerabilityInfo(
                description="Exposed .git directory revealing sensitive project history",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="Website[user=monitor]", port="HTTPS", credential="WebMonitorCredential")
                ]),
                reward_string="Found exposed .git directory revealing sensitive project history",
                cost=2.0 * weight
            ))),

    "GitHubProject": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS"),
            m.ListeningService("SSH", allowedCredentials=["StrongSSHKey"])  # "DirectoryTraversalInfo" 자격증명 허용
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=150 * weight_2,
        properties=["Ubuntu", "nginx/1.10.3", "CTFFLAG:Readme.txt-Discover secret data"],
        vulnerabilities=dict(
            #디렉토리 순회 취약점을 통해 민감한 파일에 접근 가능
            DirectoryTraversal=m.VulnerabilityInfo(
                description="Directory traversal vulnerability allowing access to sensitive files",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureResourceManager", port="HTTPS", credential="ResourceManagerAccess")
                ]),
                reward_string="Exploited directory traversal to access sensitive files",
                cost=2.0 * weight
            ))),

    "Sharepoint": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS"),
            m.ListeningService("SSH", allowedCredentials=["SharepointAccess"])
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=155 * weight_2,
        properties=["MySql", "Ubuntu", "nginx/1.10.3"],
        owned_string="Compromised user monitor with higher privileges",
        vulnerabilities=dict(
            #접근 제어의 잘못된 설정은 권한이 없는 사용자에게 접근을 허용
            MisconfiguredPermissions=m.VulnerabilityInfo(
                description="Misconfigured file permissions allowing unauthorized access",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(["AzureResourceManager[user=monitor]"]),
                reward_string="Exploited misconfigured permissions to obtain root credentials",
                cost=1.5 * weight
            ))),

    "Website[user=monitor]": m.NodeInfo(
        services=[m.ListeningService("HTTPS", allowedCredentials=["WebMonitorCredential"])
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=130 * weight_2,
        properties=["SharepointLeakingPassword"],
        vulnerabilities=dict(
            #접근 제어의 잘못된 설정은 권한이 없는 사용자에게 접근 허용 가능
            MisconfiguredAccessControl=m.VulnerabilityInfo(
                description="Misconfigured access control allowing unauthorized SharePoint access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureStorage", port="HTTPS", credential="BlobStorageToken")
                ]),
                reward_string="Exploited misconfigured access control to gain access to SharePoint credentials",
                cost=1.0 * weight
            ))), 

    "AzureResourceManager": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS", allowedCredentials=["ResourceManagerAccess"])
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=140 * weight_2,
        properties=["CTFFLAG:LeakedCustomerData2"],
        vulnerabilities=dict(
            #API 엔드포인트의 보안이 약할 때 발생할 수 있는 취약점
            InsecureAPIEndpoint=m.VulnerabilityInfo(
                description="Insecure API endpoint exposing sensitive Azure resource management operations",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureStorage", port="HTTPS", credential="BlobStorageToken")
                ]),
                reward_string="Discovered an insecure API endpoint exposing sensitive operations",
                cost=2.0 * weight
            ))),

    'AzureResourceManager[user=monitor]': m.NodeInfo(
        services=[
            m.ListeningService("HTTPS")
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=145 * weight_2,
        properties=["SensitiveAzureOperations"],
        owned_string="Sensitive Azure operations exposed due to privileged user access",
        vulnerabilities=dict(
            #고위 권한을 가진 사용자의 연산이 노출되는 취약점
            PrivilegedOperationsExposure=m.VulnerabilityInfo(
                description="Privileged operations within Azure Resource Manager exposed due to elevated user permissions",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureStorage", port="HTTPS", credential="BlobStorageToken")
                ]),
                reward_string="Gained access to privileged Azure operations and credentials due to elevated user permissions",
                cost=1.5 * weight
            ))),

    "AzureStorage": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS", allowedCredentials=["BlobStorageToken"])],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=140 * weight_2,
        properties=["CTFFLAG:LeakedCustomerData"],
        vulnerabilities=dict(
            #Blob 스토리지의 부적절한 구성으로 인한 데이터 유출 가능성
            InsecureBlobStorage=m.VulnerabilityInfo(
                description="Insecurely configured blob storage leading to data leak",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["AzureVM"]),
                reward_string="Exploited insecure blob storage configuration to leak data",
                cost=2.0 * weight
            ))),

    "AzureVM": m.NodeInfo(
        services=[
            m.ListeningService("SSH")
        ],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW)],
            outgoing=default_allow_rules
        ),
        value=140 * weight_2,
        properties=["CTFFLAG:VMPRIVATEINFO"],
        vulnerabilities=dict(
            #패치되지 않은 SSH 서비스를 통한 공격은 일반적인 보안 위협임
            UnpatchedSSHService=m.VulnerabilityInfo(
                description="Unpatched SSH service vulnerable to known exploit",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                reward_string="Exploited an unpatched SSH service to gain access to the VM",
                cost=2.0 * weight,
            ))),

    'client': m.NodeInfo(
        services=[],
        value=0,
        vulnerabilities=dict(
            #피싱은 사용자를 속여서 크레덴셜을 얻는 일반적인 방법
            PhishingVulnerability=m.VulnerabilityInfo(
                description="Susceptibility to phishing attacks leading to unauthorized access",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(["Website"]),
                reward_string="Fell for a phishing attack, leading to unauthorized access",
                cost=1.5 * weight
            )),
        agent_installed=True,
        reimagable=False),
}

global_vulnerability_library: Dict[VulnerabilityID, VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.infer_constants_from_nodes(
    cast(Iterator[Tuple[NodeID, NodeInfo]], list(nodes.items())),
    global_vulnerability_library)


def new_environment() -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=ENV_IDENTIFIERS)
