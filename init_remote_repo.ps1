cd C:\Users\OlavoDefendiDalberto\Projetos\churn-mvp

# 1) Se não for repositório, inicializa
if (-not (Test-Path .git)) {
  Write-Output "Inicializando repositório git..."
  git init
} else {
  Write-Output ".git existe — repositório já inicializado."
}

# 2) Configura nome/email (só se ainda não estiver configurado globalmente)
if (-not (git config --global user.name)) {
  git config --global user.name "Olavo Defendi Dalberto"
  git config --global user.email "seu@email"
  Write-Output "Configuração global de user.name e user.email aplicada."
} else {
  Write-Output "user.name/global já configurado."
}

# 3) Adiciona arquivos e cria commit inicial (ou commit vazio se não houver arquivos)
git add -A

# Se não existirem arquivos staged, cria um commit vazio para inicializar o repo
try {
  git commit -m "chore: inicial commit - estrutura do projeto" 2>$null
} catch {
  Write-Output "Sem arquivos para commitar. Criando commit vazio..."
  git commit --allow-empty -m "chore: initial empty commit"
}

# 4) Garante que o branch local se chame main
git branch -M main

# 5) (Re)define remote origin e faz push
# Substitua a URL abaixo pela do seu repositório remoto se for diferente.
$remoteUrl = "https://github.com/olavodd42/churn-mvp.git"
git remote remove origin 2>$null
git remote add origin $remoteUrl

# 6) Push para o remoto (vai pedir credenciais se necessário)
git push -u origin main
