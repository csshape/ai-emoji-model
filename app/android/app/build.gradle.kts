plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.framna.emojichat"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.framna.emojichat"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
}

// The demo script lives in app/fixtures, shared with the iOS app; copy it into the
// assets at build time rather than keeping a second copy that can drift.
abstract class CopyDemoConversation : DefaultTask() {
    @get:InputFile
    abstract val source: RegularFileProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        out.mkdirs()
        source.get().asFile.copyTo(out.resolve("demo_conversation.json"))
    }
}

val copyDemoConversation = tasks.register<CopyDemoConversation>("copyDemoConversation") {
    source.set(rootProject.layout.projectDirectory.file("../fixtures/demo_conversation.json"))
    outputDir.set(layout.buildDirectory.dir("generated/demoAssets"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(copyDemoConversation, CopyDemoConversation::outputDir)
    }
}

dependencies {
    implementation(project(":emojimodel"))
    implementation(libs.kotlin.serialization.json)
    implementation(libs.kotlin.coroutines.android)

    implementation(libs.androidx.core)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
