# GitHub Secrets Configuration

The following GitHub Actions secrets were created with placeholder values.
Update each secret with production credentials before running deployment workflows.

## Placeholder Secrets

### `CI_COMMIT_USER`
- Placeholder value: `ci-bot`
- Replace with: _real credential before running deployments_

### `CI_COMMIT_EMAIL`
- Placeholder value: `ci-bot@example.com`
- Replace with: _real credential before running deployments_

### `AWS_ACCESS_KEY_ID`
- Placeholder value: `PLACEHOLDER_AWS_ACCESS_KEY`
- Replace with: _real credential before running deployments_

### `AWS_SECRET_ACCESS_KEY`
- Placeholder value: `PLACEHOLDER_AWS_SECRET`
- Replace with: _real credential before running deployments_

### `AWS_REGION`
- Placeholder value: `us-east-1`
- Replace with: _real credential before running deployments_

### `TF_API_TOKEN`
- Placeholder value: `placeholder-terraform-token`
- Replace with: _real credential before running deployments_

## Updating Secrets
1. Open the repository on GitHub
2. Navigate to **Settings → Secrets and variables → Actions**
3. Edit each secret with the correct value
4. Re-run the relevant GitHub Actions workflows

Always rotate credentials regularly and use least-privilege accounts.