#!/bin/bash

# to use: type into terminal:
# chmod +x create_accs.sh
# ./create_accs.sh

array_a=("mvusd@mail.com" "mveur@mail.com" "mvgbp@mail.com"
"mvaud@mail.com" "mvbtc@mail.com" "mveth@mail.com" "mvltc@mail.com" "mvusdc@mail.com"
"mvusdt@mail.com" "mveusdt@mail.com" "mvtusdt@mail.com")

mail_no=0

# Modify array elements by appending "+mail_no" to each email address
# Check if mail_no is not equal to 0
if [ "$mail_no" -ne 0 ]; then
  # Modify array elements by appending "+mail_no" to each email address
  for i in "${!array_a[@]}"; do
    array_a[$i]="${array_a[$i]/@mail.com/+${mail_no}@mail.com}"
  done
fi

array_b=("USD" "EUR" "GBP" "AUD" "BTC" "ETH" "LTC" "USDC" "UST" "eUSDT" "tUSDT")

# Loop through the arrays
for ((i=0; i<${#array_a[@]}; i++)); do
    string_a="${array_a[i]}"
    string_b="${array_b[i]}"
    constant_string="Abcd1234 CR aq"

    # Construct and execute the command
    command="perl create_account.pl $string_a $constant_string $string_b"
    echo "Executing: $command"
    eval "$command"
done
