#!/bin/bash

##############################################################################
# MUSAN and RIR Dataset Downloader for Speaker Encoder Training Augmentation
#
# Downloads datasets used for audio augmentation:
# - MUSAN: Environmental noise, music, speech babble
# - RIRS_NOISES: Room impulse responses and point-source noises
#
# Usage:
#   ./download_augmentation_data.sh [--output-dir <path>] [--musan-only|--rir-only]
#
# Examples:
#   ./download_augmentation_data.sh --output-dir ./data
#   ./download_augmentation_data.sh --musan-only
#   ./download_augmentation_data.sh --output-dir /mnt/datasets --rir-only
##############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
OUTPUT_DIR="."
DOWNLOAD_MUSAN=true
DOWNLOAD_RIR=true

# URLS
MUSAN_URL="https://www.openslr.org/resources/17/musan.tar.gz"
RIR_URL="https://www.openslr.org/resources/28/rirs_noises.zip"

##############################################################################
# Helper Functions
##############################################################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

##############################################################################
# Parse Arguments
##############################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --musan-only)
                DOWNLOAD_MUSAN=true
                DOWNLOAD_RIR=false
                shift
                ;;
            --rir-only)
                DOWNLOAD_MUSAN=false
                DOWNLOAD_RIR=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage() {
    cat << 'EOF'
Usage: ./download_augmentation_data.sh [OPTIONS]

OPTIONS:
  --output-dir <path>    Directory to download datasets into (default: current dir)
  --musan-only          Download only MUSAN dataset
  --rir-only            Download only RIR dataset
  -h, --help            Show this help message

EXAMPLES:
  ./download_augmentation_data.sh --output-dir ./data
  ./download_augmentation_data.sh --musan-only
  ./download_augmentation_data.sh --output-dir /mnt/datasets --rir-only

DATASETS:
  MUSAN (~11GB):
    - noise/   - Environmental noise
    - music/   - Music clips
    - speech/  - Speech babble

  RIRS_NOISES (~5GB):
    - simulated_rirs/
    - real_rirs_isotropic_noises/
    - pointsource_noises/

TOTAL SIZE: ~16GB (use --musan-only or --rir-only to reduce)
EOF
}

##############################################################################
# Validation
##############################################################################

check_requirements() {
    local missing=false

    # Check for wget
    if ! command -v wget &> /dev/null; then
        print_error "wget is not installed. Please install it:"
        echo "  macOS: brew install wget"
        echo "  Ubuntu/Debian: sudo apt-get install wget"
        echo "  CentOS/RHEL: sudo yum install wget"
        missing=true
    fi

    # Check for tar (needed for MUSAN)
    if [[ "$DOWNLOAD_MUSAN" == true ]] && ! command -v tar &> /dev/null; then
        print_error "tar is not installed"
        missing=true
    fi

    # Check for unzip (needed for RIR)
    if [[ "$DOWNLOAD_RIR" == true ]] && ! command -v unzip &> /dev/null; then
        print_error "unzip is not installed. Please install it:"
        echo "  macOS: brew install unzip"
        echo "  Ubuntu/Debian: sudo apt-get install unzip"
        echo "  CentOS/RHEL: sudo yum install unzip"
        missing=true
    fi

    if [[ "$missing" == true ]]; then
        exit 1
    fi
}

check_disk_space() {
    local required_gb=16
    if [[ "$DOWNLOAD_MUSAN" == true ]] && [[ "$DOWNLOAD_RIR" == false ]]; then
        required_gb=11
    elif [[ "$DOWNLOAD_MUSAN" == false ]] && [[ "$DOWNLOAD_RIR" == true ]]; then
        required_gb=5
    fi

    # Get available space in KB and convert to GB
    local available_kb=$(df "$OUTPUT_DIR" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [[ $available_gb -lt $required_gb ]]; then
        print_warning "Limited disk space: ${available_gb}GB available, ~${required_gb}GB required"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

##############################################################################
# Download Functions
##############################################################################

download_file() {
    local url="$1"
    local output="$2"
    local filename=$(basename "$output")

    if [[ -f "$output" ]]; then
        print_warning "$filename already exists, skipping download"
        return 0
    fi

    print_info "Downloading $filename (~$(get_size_estimate "$url") GB)..."
    print_info "URL: $url"

    # Use wget with progress bar, resume support, and timeout
    if wget --show-progress \
            --progress=bar:force:noscroll \
            --continue \
            --timeout=30 \
            --tries=3 \
            -O "$output" \
            "$url"; then
        print_success "Downloaded $filename"
        return 0
    else
        print_error "Failed to download $filename"
        return 1
    fi
}

get_size_estimate() {
    case "$1" in
        *musan.tar.gz*) echo "11" ;;
        *rirs_noises.zip*) echo "5" ;;
        *) echo "?" ;;
    esac
}

##############################################################################
# Extraction Functions
##############################################################################

extract_musan() {
    local tar_file="$OUTPUT_DIR/musan.tar.gz"
    local extract_dir="$OUTPUT_DIR"

    if [[ ! -f "$tar_file" ]]; then
        print_error "MUSAN tar file not found: $tar_file"
        return 1
    fi

    if [[ -d "$extract_dir/musan" ]]; then
        print_warning "MUSAN already extracted, skipping"
        return 0
    fi

    print_info "Extracting MUSAN (~5-10 minutes)..."

    if tar -xzf "$tar_file" -C "$extract_dir"; then
        print_success "Extracted MUSAN"

        # Verify structure
        if [[ -d "$extract_dir/musan" ]]; then
            print_success "MUSAN directory verified"
            return 0
        else
            print_error "MUSAN extraction failed: directory not found"
            return 1
        fi
    else
        print_error "Failed to extract MUSAN"
        return 1
    fi
}

extract_rir() {
    local zip_file="$OUTPUT_DIR/rirs_noises.zip"
    local extract_dir="$OUTPUT_DIR"

    if [[ ! -f "$zip_file" ]]; then
        print_error "RIR zip file not found: $zip_file"
        return 1
    fi

    if [[ -d "$extract_dir/RIRS_NOISES" ]]; then
        print_warning "RIRS_NOISES already extracted, skipping"
        return 0
    fi

    print_info "Extracting RIRS_NOISES (~2-5 minutes)..."

    if unzip -q "$zip_file" -d "$extract_dir"; then
        print_success "Extracted RIRS_NOISES"

        # Verify structure
        if [[ -d "$extract_dir/RIRS_NOISES" ]]; then
            print_success "RIRS_NOISES directory verified"
            return 0
        else
            print_error "RIR extraction failed: directory not found"
            return 1
        fi
    else
        print_error "Failed to extract RIRS_NOISES"
        return 1
    fi
}

##############################################################################
# Verification Functions
##############################################################################

verify_dataset_structure() {
    print_header "Verifying Dataset Structure"

    local all_ok=true

    if [[ "$DOWNLOAD_MUSAN" == true ]]; then
        if [[ -d "$OUTPUT_DIR/musan" ]]; then
            print_success "✓ MUSAN directory found"

            local subdirs=("noise" "music" "speech")
            for subdir in "${subdirs[@]}"; do
                if [[ -d "$OUTPUT_DIR/musan/$subdir" ]]; then
                    local file_count=$(find "$OUTPUT_DIR/musan/$subdir" -type f | wc -l)
                    print_success "  ✓ musan/$subdir ($file_count files)"
                else
                    print_warning "  ⚠ musan/$subdir not found"
                    all_ok=false
                fi
            done
        else
            print_error "✗ MUSAN directory not found"
            all_ok=false
        fi
        echo
    fi

    if [[ "$DOWNLOAD_RIR" == true ]]; then
        if [[ -d "$OUTPUT_DIR/RIRS_NOISES" ]]; then
            print_success "✓ RIRS_NOISES directory found"

            local subdirs=("simulated_rirs" "real_rirs_isotropic_noises" "pointsource_noises")
            for subdir in "${subdirs[@]}"; do
                if [[ -d "$OUTPUT_DIR/RIRS_NOISES/$subdir" ]]; then
                    local file_count=$(find "$OUTPUT_DIR/RIRS_NOISES/$subdir" -type f | wc -l)
                    print_success "  ✓ RIRS_NOISES/$subdir ($file_count files)"
                else
                    print_warning "  ⚠ RIRS_NOISES/$subdir not found"
                fi
            done
        else
            print_error "✗ RIRS_NOISES directory not found"
            all_ok=false
        fi
        echo
    fi

    return $([ "$all_ok" = true ] && echo 0 || echo 1)
}

print_directory_structure() {
    print_header "Dataset Directory Structure"

    if [[ "$DOWNLOAD_MUSAN" == true ]] && [[ -d "$OUTPUT_DIR/musan" ]]; then
        echo "MUSAN ($OUTPUT_DIR/musan):"
        tree -L 2 -C "$OUTPUT_DIR/musan" 2>/dev/null || \
        find "$OUTPUT_DIR/musan" -maxdepth 2 -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g' | head -20
        echo
    fi

    if [[ "$DOWNLOAD_RIR" == true ]] && [[ -d "$OUTPUT_DIR/RIRS_NOISES" ]]; then
        echo "RIRS_NOISES ($OUTPUT_DIR/RIRS_NOISES):"
        tree -L 2 -C "$OUTPUT_DIR/RIRS_NOISES" 2>/dev/null || \
        find "$OUTPUT_DIR/RIRS_NOISES" -maxdepth 2 -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g' | head -20
        echo
    fi
}

print_summary() {
    print_header "Download Summary"

    if [[ "$DOWNLOAD_MUSAN" == true ]]; then
        echo "MUSAN:"
        echo "  Location: $OUTPUT_DIR/musan"
        echo "  Size: ~11GB"
        echo "  Contents: Environmental noise, music, speech babble"
        echo
    fi

    if [[ "$DOWNLOAD_RIR" == true ]]; then
        echo "RIRS_NOISES:"
        echo "  Location: $OUTPUT_DIR/RIRS_NOISES"
        echo "  Size: ~5GB"
        echo "  Contents: Simulated RIRs, real RIRs, point-source noises"
        echo
    fi

    local total_size=0
    [[ "$DOWNLOAD_MUSAN" == true ]] && total_size=$((total_size + 11))
    [[ "$DOWNLOAD_RIR" == true ]] && total_size=$((total_size + 5))

    echo "Total downloaded: ~${total_size}GB"
    echo
    print_success "All datasets ready for training augmentation!"
}

##############################################################################
# Cleanup Functions
##############################################################################

cleanup_on_error() {
    print_error "An error occurred. Partial downloads left in: $OUTPUT_DIR"
    echo "Run this script again to resume, or manually delete:"
    [[ -f "$OUTPUT_DIR/musan.tar.gz" ]] && echo "  rm $OUTPUT_DIR/musan.tar.gz"
    [[ -f "$OUTPUT_DIR/rirs_noises.zip" ]] && echo "  rm $OUTPUT_DIR/rirs_noises.zip"
}

##############################################################################
# Main Execution
##############################################################################

main() {
    parse_arguments "$@"

    # Validate output directory
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        print_info "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR" || {
            print_error "Failed to create directory: $OUTPUT_DIR"
            exit 1
        }
    fi

    OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)  # Convert to absolute path

    print_header "Speaker Encoder Training Augmentation Data Downloader"

    print_info "Output directory: $OUTPUT_DIR"
    print_info "MUSAN: $([ "$DOWNLOAD_MUSAN" = true ] && echo "✓ enabled" || echo "✗ disabled")"
    print_info "RIRS_NOISES: $([ "$DOWNLOAD_RIR" = true ] && echo "✓ enabled" || echo "✗ disabled")"
    echo

    # Check requirements
    check_requirements
    check_disk_space

    # Set up error handling
    trap cleanup_on_error ERR

    # Download MUSAN
    if [[ "$DOWNLOAD_MUSAN" == true ]]; then
        print_header "Downloading MUSAN"
        download_file "$MUSAN_URL" "$OUTPUT_DIR/musan.tar.gz" || exit 1
        extract_musan || exit 1
    fi

    # Download RIR
    if [[ "$DOWNLOAD_RIR" == true ]]; then
        print_header "Downloading RIRS_NOISES"
        download_file "$RIR_URL" "$OUTPUT_DIR/rirs_noises.zip" || exit 1
        extract_rir || exit 1
    fi

    # Verify and print results
    verify_dataset_structure || {
        print_warning "Some datasets may not be complete"
    }

    print_directory_structure
    print_summary
}

# Run main function with all arguments
main "$@"
