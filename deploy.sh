#!/bin/bash
echo "🚀 Deploying Trading Bot Platform to DigitalOcean"

# Get VPS IP
read -p "Enter your Droplet IP address: " VPS_IP

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf trading_bot.tar.gz \
  --exclude='venv' \
  --exclude='*.db' \
  --exclude='__pycache__' \
  --exclude='.git' \
  .

# Upload to droplet
echo "📤 Uploading to Droplet..."
scp trading_bot.tar.gz root@$VPS_IP:~/

# Upload .env separately
scp .env root@$VPS_IP:~/

# Setup script for the VPS
cat > remote_setup.sh << 'SETUP'
#!/bin/bash
echo "Setting up Trading Bot on VPS..."

# Update system
apt update && apt upgrade -y
apt install -y python3-pip python3-venv screen htop

# Extract files
mkdir -p trading_bots
cd trading_bots
tar -xzf ~/trading_bot.tar.gz
mv ~/.env .

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create start script
cat > start_all_bots.sh << 'START'
#!/bin/bash
source venv/bin/activate

# Start each bot in a screen session
screen -dmS btc_5m python run_bot.py config/btc_bb_5m.yml
echo "Started BTC 5m bot"
sleep 2

screen -dmS eth_15m python run_bot.py config/eth_bb_15m.yml
echo "Started ETH 15m bot"
sleep 2

screen -dmS btc_1h python run_bot.py config/btc_bb_1h.yml
echo "Started BTC 1h bot"

echo "All bots started! Use 'screen -ls' to see them"
START
chmod +x start_all_bots.sh

# Create monitoring script
cat > monitor.sh << 'MONITOR'
#!/bin/bash
source venv/bin/activate
python check_bots.py
MONITOR
chmod +x monitor.sh

echo "✅ Setup complete!"
SETUP

# Upload and run setup
scp remote_setup.sh root@$VPS_IP:~/
ssh root@$VPS_IP "bash remote_setup.sh"

echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. SSH into your droplet: ssh root@$VPS_IP"
echo "2. cd trading_bots"
echo "3. Start all bots: ./start_all_bots.sh"
echo "4. Start dashboard: screen -dmS dashboard streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
echo "5. View dashboard at: http://$VPS_IP:8501"
