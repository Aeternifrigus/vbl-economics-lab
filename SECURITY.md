# Security

- Report vulnerabilities privately through GitHub's "Report a vulnerability" button on the Security tab,
  not in a public issue.
- Secrets: the only credential is `FRED_API_KEY`. It is read from the environment or an untracked `.env`
  file and is never written to logs or snapshots. In CI it lives in a repository secret. If a key is ever
  committed, revoke it at fred.stlouisfed.org and issue a new one.
- Dependencies are pinned with hashes in `requirements.lock` and `requirements-dev.lock` and updated
  through Dependabot pull requests.
