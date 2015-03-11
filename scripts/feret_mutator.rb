#!/usr/bin/env ruby
require 'json'
require 'fileutils'

class FeretMutator
  include FileUtils::Verbose

  IMAGE_POSES = [
    'fa',
    'fb',
    'rb',
    'rc'
  ].freeze

  IMAGE_GLOB = "*{#{IMAGE_POSES.join(',')}}*.ppm.bz2"

  attr_reader :inpath
  attr_reader :outpath
  attr_reader :images_dir
  attr_reader :grounds_dir

  def initialize(inpath, outpath)
    raise 'WTF? Give me an inpath & outpath pls' unless inpath && outpath

    @inpath   = inpath
    @outpath  = outpath
    @manifest = []
  end

  def mutate
    mkdir(outpath) unless File.exist?(outpath)

    [ 'dvd1', 'dvd2' ].each do |dvd_dir|
      data_dir     = File.join(inpath, dvd_dir, 'data')
      @images_dir  = File.join(data_dir, 'smaller')
      @grounds_dir = File.join(data_dir, 'ground_truths', 'name_value')

      mutate_images
    end

    File.open(File.join(outpath, 'manifest.json'), 'w') do |f|
      f.write(JSON.dump(@manifest))
    end
  end

  private
  def mutate_images
    Dir.glob("#{images_dir}/*").each do |label_dir|
      label      = File.basename(label_dir)
      outdir     = File.join(outpath, label)
      image_glob = "#{label_dir}/#{IMAGE_GLOB}"

      truths           = parse_grounds(File.join(grounds_dir, label, "#{label}.txt"))
      truths['images'] = []


      mkdir outdir unless File.exist?(outdir)

      Dir.glob(image_glob).each do |image_filepath|
        image_l  = File.basename(image_filepath).split('.').first
        ground_f = File.join(grounds_dir, label, "#{image_l}.txt")
        truths['images'] << parse_grounds(ground_f)
        copy_and_convert_image(label, image_filepath)
      end

      File.open(File.join(outdir, 'truths.json'), 'w') do |f|
        f.write(JSON.pretty_generate(truths))
      end

      @manifest << truths
    end
  end

  def parse_grounds(path)
    {}.tap do |hash|
      File.read(path).split("\n").each do |line|
        k,v = line.split('=')

        next if [ 'compression' ].include?(k)

        if k == 'relative'
          v = File.basename(v).split('.').first
          v = "#{v}.png"
          k = 'filename'
        end

        if k == 'format'
          v = 'png'
        end

        hash[k] = v
      end
    end
  end

  def copy_and_convert_image(label, filepath)
    basefile  = File.basename(filepath).split('.').first
    converted = File.join(outpath, label, "#{basefile}.png")
    zipped    = File.join(outpath, label, "#{basefile}.ppm.bz2")
    unzipped  = File.join(outpath, label, "#{basefile}.ppm")

    cp(filepath, zipped) unless File.exist?(zipped)
    system("bzip2 -d #{zipped}")
    system("convert #{unzipped} #{converted}")
    rm(unzipped)
  end
end

if __FILE__ == $0
  FeretMutator.new(ARGV[0], ARGV[1]).mutate
end
